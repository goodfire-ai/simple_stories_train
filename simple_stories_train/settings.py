"""Global path settings for simple_stories_train."""

from pathlib import Path

TRAIN_OUT_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd/target_models")
SLURM_LOG_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs")
REPO_ROOT = Path(__file__).parent.parent.resolve()
