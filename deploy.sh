#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=h200-reserved-default
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --job-name=spd-train-48h
#SBATCH --output=/mnt/polished-lake/home/braun/slurm_logs/slurm-%j.out

if [ -z "$SLURM_JOB_ID" ]; then
    # Running locally - submit to SLURM and print log path
    JOB_ID=$(sbatch --parsable "$0" "$@")
    echo "Submitted job $JOB_ID"
    echo "Log file: /mnt/polished-lake/home/braun/slurm_logs/slurm-${JOB_ID}.out"
    exit 0
fi

cd /mnt/polished-lake/home/braun/gf-simple_stories_train
# python simple_stories_train/train.py simple_stories_train/train_config.yaml
torchrun --standalone --nproc_per_node=8 simple_stories_train/train.py simple_stories_train/train_config_pile.yaml