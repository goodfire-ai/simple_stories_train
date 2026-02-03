#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=ss-train-20m
#SBATCH --output=slurm_logs/slurm-%A.log
#SBATCH --error=slurm_logs/slurm-%A.log

set -euo pipefail

# Run the script
python simple_stories_train/train.py simple_stories_train/train_config.yaml