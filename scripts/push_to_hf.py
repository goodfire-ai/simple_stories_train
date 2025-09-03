"""
Usage:
```bash
python scripts/push_to_hf.py \
  --checkpoint-path /path/to/checkpoint.pt \
  --repo-id your-username/your-repo \
  --token $HF_TOKEN
```
"""

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from transformers import PreTrainedModel

from simple_stories_train.models.gpt2 import (
    GPT2,
    GPT2Config,
    convert_gpt2_to_hf_gpt2,
)
from simple_stories_train.models.llama import (
    Llama,
    LlamaConfig,
    convert_llama_to_llama_for_causal_lm,
)
from simple_stories_train.models.model_configs import MODEL_CONFIGS


@dataclass
class PushArgs:
    checkpoint_path: Path
    repo_id: str
    token: str | None
    private: bool
    revision: str | None
    commit_message: str | None
    model_card_readme: Path | None


def parse_args() -> PushArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local custom Llama checkpoint, convert to Hugging Face format, and "
            "push to the Hub."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the local .pt checkpoint saved via torch.save(model.state_dict(), ...)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination repository in the form 'username/repo_name' or 'org/repo_name'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, will use HF_TOKEN env var if present.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private (default: public).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch name on the Hub (e.g., 'main').",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--model-card-readme",
        type=str,
        default=None,
        help="Optional path to a README.md to upload as the model card.",
    )

    ns = parser.parse_args()
    token = ns.token or os.environ.get("HF_TOKEN")

    return PushArgs(
        checkpoint_path=Path(ns.checkpoint_path).expanduser().resolve(),
        repo_id=ns.repo_id,
        token=token,
        private=bool(ns.private),
        revision=ns.revision,
        commit_message=ns.commit_message,
        model_card_readme=(
            Path(ns.model_card_readme).expanduser().resolve() if ns.model_card_readme else None
        ),
    )


def load_config_from_checkpoint_dir(checkpoint_path: Path) -> tuple[str, LlamaConfig | GPT2Config]:
    """Load model config by reading model_id from final_config.yaml adjacent to the checkpoint.

    Returns (model_id, config) where config is one of LlamaConfig or GPT2Config.
    """
    final_cfg_path = checkpoint_path.parent / "final_config.yaml"
    if not final_cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find 'final_config.yaml' next to checkpoint at {final_cfg_path}"
        )

    with final_cfg_path.open("r") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    model_id = data.get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("'model_id' missing or invalid in final_config.yaml")

    preset = MODEL_CONFIGS.get(model_id)
    assert preset is None or isinstance(preset, LlamaConfig | GPT2Config)
    if preset is None:
        raise ValueError(
            f"Unknown model_id '{model_id}' in final_config.yaml."
            f" Available: {tuple(MODEL_CONFIGS.keys())}"
        )

    # Optionally override context length from training config if present
    train_ds_cfg = data.get("train_dataset_config", {}) or {}
    n_ctx_override = train_ds_cfg.get("n_ctx")
    if isinstance(n_ctx_override, int) and n_ctx_override > 0:
        if isinstance(preset, LlamaConfig):
            return model_id, preset.model_copy(
                update={"n_ctx": n_ctx_override, "block_size": n_ctx_override}
            )
        if isinstance(preset, GPT2Config):
            return model_id, preset.model_copy(update={"block_size": n_ctx_override})
    return model_id, preset


def load_custom_model(
    checkpoint_path: Path, model_id: str, config: LlamaConfig | GPT2Config
) -> Llama | GPT2:
    # Llama requires special loader to rebuild rotary buffers
    if isinstance(config, LlamaConfig):
        model = Llama.from_pretrained(str(checkpoint_path), config=config, strict=True)
    else:
        # GPT-2: regular state_dict load
        state_dict = torch.load(str(checkpoint_path), weights_only=True, map_location="cpu")
        # Strip DDP prefixes if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model = GPT2(config)
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def convert_to_hf_model(custom_model: Llama | GPT2) -> PreTrainedModel:
    if isinstance(custom_model, Llama):
        hf_model = convert_llama_to_llama_for_causal_lm(custom_model)
    else:
        hf_model = convert_gpt2_to_hf_gpt2(custom_model)
    hf_model.eval()
    return hf_model


def _resolve_tokenizer_path(final_cfg_path: Path) -> Path | None:
    """Try to resolve a tokenizer file path from the final_config.yaml next to the checkpoint.

    Returns absolute path to the tokenizer json if it can be found, otherwise None.

    TODO: Save the tokenizer when training the model.
    """
    try:
        with final_cfg_path.open("r") as f:
            data: dict[str, Any] = yaml.safe_load(f)
    except Exception:
        return None

    train_ds_cfg = data.get("train_dataset_config", {}) or {}
    tokenizer_rel: str | None = train_ds_cfg.get("tokenizer_file_path")
    if not tokenizer_rel or not isinstance(tokenizer_rel, str):
        return None

    # As a last resort, if the file name matches a known tokenizer in the repo, use it
    known_default = Path("simple_stories_train/tokenizer/simplestories-tokenizer.json")
    if known_default.is_file():
        return known_default.resolve()

    return None


def upload_tokenizer_to_hub(
    repo_id: str,
    token: str | None,
    model_max_length: int | None,
    checkpoint_path: Path,
) -> None:
    """Upload tokenizer artifacts (minimal set) to the Hub model repo.

    Uploads:
      - tokenizer.json (raw Tokenizers file)
      - tokenizer_config.json (minimal, includes eos/unk tokens and max length if known)
    """
    final_cfg_path = checkpoint_path.parent / "final_config.yaml"
    tokenizer_path = _resolve_tokenizer_path(final_cfg_path)
    if tokenizer_path is None or not tokenizer_path.exists():
        # Nothing to upload
        return

    api = HfApi()

    # Upload tokenizer.json (rename if needed)
    api.upload_file(
        path_or_fileobj=str(tokenizer_path),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    # Build tokenizer_config.json matching desired structure
    # Discover IDs for special tokens from the tokenizer file
    unk_token = "[UNK]"
    eos_token = "[EOS]"
    added_tokens_decoder: dict[str, dict[str, Any]] = {}

    try:
        tk: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
        unk_id = tk.token_to_id(unk_token)
        eos_id = tk.token_to_id(eos_token)
    except Exception:
        unk_id = None
        eos_id = None

    def _entry(content: str) -> dict[str, Any]:
        return {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    if isinstance(unk_id, int):
        added_tokens_decoder[str(unk_id)] = _entry(unk_token)
    if isinstance(eos_id, int):
        added_tokens_decoder[str(eos_id)] = _entry(eos_token)

    # Use HF's sentinel for unlimited length to mirror common configs
    unlimited_len = int(1e30)

    cfg: dict[str, Any] = {
        "added_tokens_decoder": added_tokens_decoder,
        "clean_up_tokenization_spaces": False,
        "eos_token": eos_token,
        "extra_special_tokens": {},
        "model_max_length": unlimited_len,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": unk_token,
    }

    cfg_bytes = json.dumps(cfg, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(cfg_bytes),
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def push_model_to_hub(
    hf_model: PreTrainedModel,
    repo_id: str,
    token: str | None,
    private: bool,
    revision: str | None,
    commit_message: str | None,
) -> None:
    # Call via the class to satisfy certain linters complaining about 'self'
    hf_model.__class__.push_to_hub(
        hf_model,
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
        revision=revision,
    )


def optionally_upload_readme(repo_id: str, token: str | None, readme_path: Path | None) -> None:
    if readme_path is None:
        return
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def main() -> None:
    args = parse_args()

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    model_id, config = load_config_from_checkpoint_dir(args.checkpoint_path)
    custom_model = load_custom_model(args.checkpoint_path, model_id, config)

    # Convert and push
    hf_model = convert_to_hf_model(custom_model)
    push_model_to_hub(
        hf_model=hf_model,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        revision=args.revision,
        commit_message=args.commit_message,
    )

    # Upload tokenizer artifacts (minimal set)
    model_max_len: int | None = None
    if isinstance(config, LlamaConfig):
        model_max_len = config.n_ctx
    elif isinstance(config, GPT2Config):
        model_max_len = config.block_size
    upload_tokenizer_to_hub(
        repo_id=args.repo_id,
        token=args.token,
        model_max_length=model_max_len,
        checkpoint_path=args.checkpoint_path,
    )

    # Optional README
    optionally_upload_readme(args.repo_id, args.token, args.model_card_readme)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
