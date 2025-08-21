from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import yaml

from simple_stories_train.utils import REPO_ROOT

WANDB_PATH_PREFIX = "wandb:"


def _is_wandb_path(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX)


def _wandb_slug(path_no_prefix: str) -> str:
    return path_no_prefix.strip("/").replace("/", "__")


def _cache_dir_for_slug(slug: str) -> Path:
    return REPO_ROOT / ".cache" / "wandb_runs" / slug


def _download_wandb_files(wandb_path_no_prefix: str) -> tuple[Path, Path, Path]:
    """Download 'model_step_*.pt', 'final_config.yaml' and 'model_config.yaml' for the given W&B run.

    Returns (checkpoint_path, config_path, model_config_path).
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

    # Download both
    config_file.download(root=str(cache_dir), replace=True)
    model_config_file.download(root=str(cache_dir), replace=True)
    latest_ckpt_file.download(root=str(cache_dir), replace=True)

    ckpt_path = cache_dir / latest_ckpt_file.name
    config_path = cache_dir / config_file.name
    model_config_path = cache_dir / model_config_file.name
    return ckpt_path, config_path, model_config_path


@dataclass
class RunInfo:
    """Run info from training a model in this repository.

    NOTE: We just return dicts instead of config objects to avoid circular imports. A refactor might
    avoid this.
    """

    checkpoint_path: Path
    config_dict: dict[str, Any]
    model_config_dict: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> RunInfo:
        """Load run info from a W&B run string or a local path.

        - W&B strings: 'wandb:goodfire/spd-play/runs/152i5k4r'
        - Local: path to a checkpoint file or a directory containing checkpoints
        """
        # Handle W&B path
        if _is_wandb_path(path):
            wandb_path = path[len(WANDB_PATH_PREFIX) :]  # type: ignore[index]
            ckpt_path, config_path, model_config_path = _download_wandb_files(wandb_path)
        else:
            ckpt_path = Path(path)
            assert ckpt_path.is_file(), f"Expected a file, got {ckpt_path}"
            config_path = ckpt_path.parent / "final_config.yaml"
            assert config_path.exists(), (
                f"Expected config at {config_path} next to checkpoint {ckpt_path}"
            )
            model_config_path = ckpt_path.parent / "model_config.yaml"
            assert model_config_path.exists(), (
                f"Expected model config at {model_config_path} next to checkpoint {ckpt_path}"
            )

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        with open(model_config_path) as f:
            model_config_dict = yaml.safe_load(f)

        return cls(
            checkpoint_path=ckpt_path,
            config_dict=config_dict,
            model_config_dict=model_config_dict,
        )
