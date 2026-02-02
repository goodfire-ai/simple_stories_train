#!/usr/bin/env python3
"""Deploy script for training jobs.

This script can be run directly or via the `sst-train` command after installing the package.

Usage:
    # Submit to SLURM
    sst-train --config_path configs/llama_simple_mlp_4L_wide.yaml

    # Submit with custom GPU count and partition
    sst-train --config_path ... --n_gpus 4 --partition h200-reserved-default

    # Run locally (no SLURM)
    sst-train --config_path ... --local

    # Pass additional arguments to train.py
    sst-train --config_path ... --num_iterations 1000 --learning_rate 5e-5

    # Alternative: run the script directly
    python scripts/deploy.py --config_path configs/llama_simple_mlp_4L_wide.yaml
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
SLURM_LOG_DIR = Path(os.environ.get("HOME", "/tmp")) / "slurm_logs"


def create_slurm_script(
    config_path: Path,
    n_gpus: int,
    partition: str,
    time_limit: str,
    job_name: str,
    extra_args: list[str],
) -> str:
    """Generate SLURM batch script content."""
    # Build the training command
    train_args = [str(config_path)] + extra_args
    train_cmd = (
        f"torchrun --standalone --nproc_per_node={n_gpus} "
        f"-m simple_stories_train.train {' '.join(train_args)}"
    )

    script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --time={time_limit}
#SBATCH --job-name={job_name}
#SBATCH --output={SLURM_LOG_DIR}/slurm-%j.out

cd {REPO_ROOT}
{train_cmd}
"""
    return script


def run_local(config_path: Path, n_gpus: int, extra_args: list[str]) -> None:
    """Run training locally with torchrun."""
    train_args = [str(config_path)] + extra_args

    if n_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={n_gpus}",
            "-m",
            "simple_stories_train.train",
        ] + train_args
    else:
        cmd = [sys.executable, "-m", "simple_stories_train.train"] + train_args

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def submit_slurm(
    config_path: Path,
    n_gpus: int,
    partition: str,
    time_limit: str,
    job_name: str,
    extra_args: list[str],
) -> None:
    """Submit job to SLURM."""
    # Ensure log directory exists
    SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    script_content = create_slurm_script(
        config_path=config_path,
        n_gpus=n_gpus,
        partition=partition,
        time_limit=time_limit,
        job_name=job_name,
        extra_args=extra_args,
    )

    # Write script to temp file and submit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip()
        print(f"Submitted job {job_id}")
        print(f"Log file: {SLURM_LOG_DIR}/slurm-{job_id}.out")
    finally:
        os.unlink(script_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy training jobs to SLURM or run locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the unified config YAML file",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: 8 for SLURM, 1 for local)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="h200-reserved-default",
        help="SLURM partition (default: h200-reserved-default)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="72:00:00",
        help="SLURM time limit (default: 72:00:00)",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="SLURM job name (default: derived from config file name)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of submitting to SLURM",
    )

    # Parse known args to allow passing extra args to train.py
    args, extra_args = parser.parse_known_args()

    # Validate config path
    if not args.config_path.exists():
        print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
        sys.exit(1)

    # Default job name from config file
    job_name = args.job_name or args.config_path.stem

    # Set default n_gpus based on mode
    n_gpus = args.n_gpus
    if n_gpus is None:
        n_gpus = 1 if args.local else 8

    if args.local:
        run_local(
            config_path=args.config_path,
            n_gpus=n_gpus,
            extra_args=extra_args,
        )
    else:
        submit_slurm(
            config_path=args.config_path,
            n_gpus=n_gpus,
            partition=args.partition,
            time_limit=args.time,
            job_name=job_name,
            extra_args=extra_args,
        )


if __name__ == "__main__":
    main()
