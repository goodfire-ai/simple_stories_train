from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import yaml
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

from simple_stories_train.utils import REPO_ROOT

WANDB_PATH_PREFIX = "wandb:"


def _is_wandb_path(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX)


def _wandb_slug(path_no_prefix: str) -> str:
    return path_no_prefix.strip("/").replace("/", "__")


def _cache_dir_for_slug(slug: str) -> Path:
    return REPO_ROOT / ".cache" / "wandb_runs" / slug


def _download_wandb_files(
    wandb_path_no_prefix: str,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Download core artifacts for the given W&B run.

    Returns (checkpoint_path, config_path, model_config_path, ln_stds_path, tokenizer_path).
    """
    slug = _wandb_slug(wandb_path_no_prefix)
    cache_dir = _cache_dir_for_slug(slug)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(wandb_path_no_prefix)
    files = list(run.files())

    # Locate config file
    config_file = None
    for f in files:
        if f.name.endswith("final_config.yaml"):
            config_file = f
            break
    if config_file is None:
        raise FileNotFoundError("Could not find 'final_config.yaml' in the W&B run files.")

    model_config_file = None
    for f in files:
        if f.name.endswith("model_config.yaml"):
            model_config_file = f
            break
    if model_config_file is None:
        raise FileNotFoundError("Could not find 'model_config.yaml' in the W&B run files.")

    ln_stds_file = None
    for f in files:
        if f.name.endswith("ln-stds.json"):
            ln_stds_file = f
            break

    tokenizer_file = None
    for f in files:
        if f.name.endswith("tokenizer.json"):
            tokenizer_file = f
            break

    # Locate latest checkpoint by step number
    step_re = re.compile(r"^model_step_(\d+)\.pt$")
    ckpt_candidates: list[tuple[int, Any]] = []
    for f in files:
        m = step_re.match(f.name)
        if m:
            ckpt_candidates.append((int(m.group(1)), f))
    if not ckpt_candidates:
        raise FileNotFoundError(
            "Could not find any 'model_step_*.pt' checkpoint in the W&B run files."
        )
    ckpt_candidates.sort(key=lambda t: t[0])
    _, latest_ckpt_file = ckpt_candidates[-1]

    config_file.download(root=str(cache_dir), replace=True)
    model_config_file.download(root=str(cache_dir), replace=True)
    latest_ckpt_file.download(root=str(cache_dir), replace=True)
    if ln_stds_file is not None:
        ln_stds_file.download(root=str(cache_dir), replace=True)
    if tokenizer_file is not None:
        tokenizer_file.download(root=str(cache_dir), replace=True)

    ckpt_path = cache_dir / latest_ckpt_file.name
    config_path = cache_dir / config_file.name
    model_config_path = cache_dir / model_config_file.name
    ln_stds_path = cache_dir / ln_stds_file.name if ln_stds_file is not None else None
    tokenizer_path = cache_dir / tokenizer_file.name if tokenizer_file is not None else None
    return ckpt_path, config_path, model_config_path, ln_stds_path, tokenizer_path


@dataclass
class RunInfo:
    """Run info from training a model in this repository.

    NOTE: We just return dicts instead of config objects to avoid circular imports. A refactor might
    avoid this.
    """

    checkpoint_path: Path
    config_dict: dict[str, Any]
    model_config_dict: dict[str, Any]
    ln_stds: dict[str, float] | None
    tokenizer_path: Path | None
    hf_tokenizer_path: str | None

    @classmethod
    def from_path(cls, path: str | Path) -> RunInfo:
        """Load run info from a W&B run string or a local path.

        - W&B strings: 'wandb:goodfire/spd-play/runs/152i5k4r'
        - Local: path to a checkpoint file or a directory containing checkpoints
        """
        # Handle W&B path
        if _is_wandb_path(path):
            wandb_path = path[len(WANDB_PATH_PREFIX) :]  # type: ignore[index]
            (
                ckpt_path,
                config_path,
                model_config_path,
                ln_stds_path,
                tokenizer_path,
            ) = _download_wandb_files(wandb_path)
        else:
            ckpt_path = Path(path)
            assert ckpt_path.is_file(), f"Expected a file, got {ckpt_path}"
            # Look for configs and tokenizer in parent.parent (output_dir)
            output_dir = ckpt_path.parent.parent
            config_path = output_dir / "final_config.yaml"
            model_config_path = output_dir / "model_config.yaml"
            assert config_path.exists(), (
                f"Expected config at {config_path} next to checkpoint {ckpt_path}"
            )
            assert model_config_path.exists(), (
                f"Expected model config at {model_config_path} next to checkpoint {ckpt_path}"
            )
            tokenizer_path = output_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                tokenizer_path = None
            ln_stds_path = None  # Not supported for local paths

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        with open(model_config_path) as f:
            model_config_dict = yaml.safe_load(f)

        if ln_stds_path is not None:
            with open(ln_stds_path) as f:
                ln_stds = json.load(f)
        else:
            ln_stds = None

        # Optional HF tokenizer reference from config
        try:
            hf_tok_path = config_dict["train_dataset_config"]["hf_tokenizer_path"]
            hf_tok_path = hf_tok_path if isinstance(hf_tok_path, str) else None
        except Exception:
            hf_tok_path = None

        return cls(
            checkpoint_path=ckpt_path,
            config_dict=config_dict,
            model_config_dict=model_config_dict,
            ln_stds=ln_stds,
            tokenizer_path=tokenizer_path,
            hf_tokenizer_path=hf_tok_path,
        )

    def load_tokenizer(self) -> HFTokenizer:
        """Load tokenizer with simple HF/local logic like in dataloaders.py."""
        # Prefer HF path if specified
        if self.hf_tokenizer_path is not None:
            return AutoTokenizer.from_pretrained(
                self.hf_tokenizer_path,
                add_bos_token=False,
                unk_token="[UNK]",
                eos_token="[EOS]",
                bos_token=None,
            ).backend_tokenizer

        # Next, prefer tokenizer.json adjacent to outputs (downloaded from wandb or local)
        if self.tokenizer_path is not None and self.tokenizer_path.exists():
            return HFTokenizer.from_file(str(self.tokenizer_path))

        # Finally, fall back to tokenizer_file_path from config if present
        try:
            tok_cfg = self.config_dict["train_dataset_config"]["tokenizer_file_path"]
            if isinstance(tok_cfg, str):
                p = Path(tok_cfg)
                if p.is_file():
                    return HFTokenizer.from_file(str(p))
                # try in output_dir (checkpoint.parent.parent)
                cand = self.checkpoint_path.parent.parent / p.name
                if cand.is_file():
                    return HFTokenizer.from_file(str(cand))
        except Exception:
            pass

        raise FileNotFoundError("Could not resolve a tokenizer for this RunInfo")
