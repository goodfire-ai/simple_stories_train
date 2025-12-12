# simple_stories_train

Training framework for small language models using SimpleStories, a large-scale synthetic dataset of over 2 million short stories in simple language.

**Paper:** [Parameterized Synthetic Text Generation with SimpleStories](https://arxiv.org/abs/2504.09184)  
**Models & Dataset:** [🤗 SimpleStories on Hugging Face](https://huggingface.co/SimpleStories)

_Note: This implementation removes the morphological analysis functionality described in the paper (page 5), where common English affixes (prefixes like "un", "re" and suffixes like "ed", "ing", "ly") were included as part of the tokenizer's initial alphabet. Empirical testing showed the WordPiece trainer naturally discovers these morphemes during training, making explicit seeding redundant._

## Installation

From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```

## Usage

### Training a model
```bash
python -m simple_stories_train.train [PATH/TO/CONFIG.yaml] [--key1 value1 --key2 value2 ...]
```
where
- `PATH/TO/CONFIG.yaml` contains the training config. If no path is provided, a default config will be used.
- `--key1 value1 --key2 value2 ...` override values in the config. Note that if you wish to update a
  nested value, you must use dotted notation (e.g. `--train_dataset_config.name my_dataset`).

If running on CPU, you may need to set `--compile=False`.

To run on multiple GPUs, use
```
torchrun --standalone --nproc_per_node=N -m simple_stories_train.train ...
```
where `N` is the number of GPUs to use.

### SLURM Cluster Submission

To submit training jobs to a SLURM cluster:

```bash
# Generate SLURM script (review before submitting)
python scripts/slurm_train.py --config simple_stories_train/train_config.yaml

# Generate and submit (8 GPUs, single node)
python scripts/slurm_train.py --config simple_stories_train/train_config.yaml --n_gpus 8 --submit

# Multi-node training (16 GPUs = 2 nodes × 8 GPUs)
python scripts/slurm_train.py --config simple_stories_train/train_config.yaml --n_gpus 16 --submit

# Custom partition and time limit
python scripts/slurm_train.py --config simple_stories_train/train_config.yaml --partition gpu --time 24:00:00 --submit
```

Options:
- `--config`: Path to training config YAML (required)
- `--n_gpus`: Number of GPUs (default: 8). Values >8 trigger multi-node DDP.
- `--partition`: SLURM partition name (default: 'gpu')
- `--time`: Job time limit in HH:MM:SS (default: '72:00:00')
- `--job_name`: Custom job name (default: auto-generated from config)
- `--submit`: Actually submit the job (default: just generate script)

Generated scripts are saved to `slurm_scripts/` and logs to `slurm_logs/`.

### Logging with Weights & Biases
To track training with Weights & Biases, you can set the WANDB_PROJECT and WANDB_API_KEY variables in
`.env`. API keys can be obtained from your [Weights & Biases account settings](https://wandb.ai/settings).

## Acknowledgments

- Training script is based on the efficient [train_gpt2.py](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) in [llm.c](https://github.com/karpathy/llm.c) (licensed under MIT ((c) 2024 Andrej Karpathy))
- Some model architecture implementations are based on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (licensed under MIT ((c) 2022 TransformerLensOrg))
