"""SLURM submission script for simple_stories_train.

Usage:
    # Generate SLURM script only (don't submit)
    python scripts/slurm_train.py --config simple_stories_train/train_config_pile.yaml

    # Generate and submit
    python scripts/slurm_train.py --config simple_stories_train/train_config_pile.yaml --submit

    # Multi-GPU (single node, 8 GPUs)
    python scripts/slurm_train.py --config simple_stories_train/train_config_pile.yaml --n_gpus 8 --submit

    # Multi-node (16 GPUs = 2 nodes x 8 GPUs)
    python scripts/slurm_train.py --config simple_stories_train/train_config_pile.yaml --n_gpus 16 --submit

    # Custom partition and time
    python scripts/slurm_train.py --config simple_stories_train/train_config_pile.yaml --partition gpu --time 24:00:00
"""

import shlex
import subprocess
import uuid
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).parent.parent.resolve()
GPUS_PER_NODE = 8

CUDA_FLAGS = {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


def _choose_master_port(run_id: str) -> int:
    """Choose a unique port for DDP master.

    Uses a stable hash of run_id mapped into a high, unprivileged port range.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(run_id.encode()).hexdigest(), 16)
    return base + (h % span)


def create_slurm_script(
    config_path: str,
    n_gpus: int,
    partition: str,
    time_limit: str,
    job_name: str,
    run_id: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Create a SLURM submission script.

    Args:
        config_path: Path to training config YAML file
        n_gpus: Number of GPUs (1-8 for single node, >8 for multi-node)
        partition: SLURM partition name
        time_limit: Job time limit (HH:MM:SS format)
        job_name: SLURM job name
        run_id: Unique run identifier
        overrides: Optional config overrides as key=value pairs
    """
    port = _choose_master_port(run_id)

    # Build config overrides string
    override_args = ""
    if overrides:
        for key, value in overrides.items():
            override_args += f" --{key} {shlex.quote(str(value))}"

    # Build the training command based on GPU count
    if n_gpus == 1:
        train_cmd = f"python -m simple_stories_train.train {config_path}{override_args}"
    elif n_gpus <= GPUS_PER_NODE:
        # Single-node DDP
        train_cmd = (
            f"torchrun --standalone --nproc_per_node={n_gpus} --master_port={port} "
            f"-m simple_stories_train.train {config_path}{override_args}"
        )
    else:
        # Multi-node DDP via srun + torchrun
        n_nodes = n_gpus // GPUS_PER_NODE
        if n_gpus % GPUS_PER_NODE != 0:
            raise ValueError(f"n_gpus ({n_gpus}) must be divisible by {GPUS_PER_NODE} for multi-node")

        torchrun_cmd = (
            f"torchrun "
            f"--nnodes={n_nodes} "
            f"--node_rank=$SLURM_PROCID "
            f"--nproc_per_node={GPUS_PER_NODE} "
            f'--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
            f"--master_port={port} "
            f"-m simple_stories_train.train {config_path}{override_args}"
        )
        # Wrap in srun bash -c for proper variable expansion
        train_cmd = f"srun --cpus-per-task=128 bash -c {shlex.quote(torchrun_cmd)}"

    # Compute SLURM resource allocation
    if n_gpus <= GPUS_PER_NODE:
        n_nodes = 1
        gpus_per_task = n_gpus
    else:
        n_nodes = n_gpus // GPUS_PER_NODE
        gpus_per_task = GPUS_PER_NODE

    # Build environment exports
    env_exports = "\n".join(f"export {k}={v}" for k, v in CUDA_FLAGS.items())

    logs_dir = REPO_ROOT / "slurm_logs"
    script = f"""\
#!/bin/bash
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks={n_nodes}
#SBATCH --gres=gpu:{gpus_per_task}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/slurm-%j.out
#SBATCH --error={logs_dir}/slurm-%j.err

echo "========================================"
echo "SLURM Job Info"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: {gpus_per_task}"
echo "Total GPUs: {n_gpus}"
echo "Run ID: {run_id}"
echo "========================================"

# Set up environment
{env_exports}

# Change to repository directory
cd {REPO_ROOT}

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Debug info for multi-node
echo "Debug: SLURM_NODEID=$SLURM_NODEID"
echo "Debug: SLURM_PROCID=$SLURM_PROCID"
echo "Debug: SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
if [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "Debug: Master node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
fi

echo "========================================"
echo "Starting training..."
echo "========================================"

{train_cmd}

echo "========================================"
echo "Training complete"
echo "========================================"
"""
    return script


def main(
    config: str,
    n_gpus: int = 8,
    partition: str = "h200-reserved-default",
    time: str = "72:00:00",
    job_name: str | None = None,
    submit: bool = False,
    **overrides: str,
) -> None:
    """Generate and optionally submit a SLURM training job.

    Args:
        config: Path to training config YAML file (required)
        n_gpus: Number of GPUs to use (default: 8)
        partition: SLURM partition (default: 'h200-reserved-default')
        time: Time limit in HH:MM:SS format (default: '72:00:00')
        job_name: Optional job name (default: auto-generated from config)
        submit: If True, submit the job via sbatch (default: False)
        **overrides: Additional config overrides (e.g., --learning_rate 1e-3)
    """
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Generate unique run ID
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Auto-generate job name from config if not provided
    if job_name is None:
        job_name = f"sst_{config_path.stem}"

    # Create logs directory
    logs_dir = REPO_ROOT / "slurm_logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate the script
    script_content = create_slurm_script(
        config_path=str(config_path),
        n_gpus=n_gpus,
        partition=partition,
        time_limit=time,
        job_name=job_name,
        run_id=run_id,
        overrides=overrides if overrides else None,
    )

    # Write script to file
    scripts_dir = REPO_ROOT / "slurm_scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_path = scripts_dir / f"{job_name}_{run_id}.sh"
    script_path.write_text(script_content)
    print(f"Generated SLURM script: {script_path}")

    if submit:
        # Submit via sbatch
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted SLURM job: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
        print(f"View logs: tail -f {logs_dir}/slurm-{job_id}.out")
    else:
        print("\nTo submit this job, run:")
        print(f"  sbatch {script_path}")
        print("\nOr re-run with --submit flag:")
        print(f"  python scripts/slurm_train.py --config {config} --n_gpus {n_gpus} --submit")


if __name__ == "__main__":
    fire.Fire(main)
