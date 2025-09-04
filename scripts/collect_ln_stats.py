"""Script to collect LN stats for a model.

Usage:
```bash
python scripts/collect_ln_stats.py --batches 50 --batch_size 64 --n_ctx 512 --model_path wandb:goodfire/spd/runs/syhzse3u
```
"""

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import wandb
from jaxtyping import Float
from torch import Tensor

from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from simple_stories_train.run_info import RunInfo
from simple_stories_train.utils import REPO_ROOT


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=50, help="Number of batches to average")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per step")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--wandb_path",
        type=str,
        required=True,
        help="wandb path (e.g. wandb:goodfire/spd/runs/syhzse3u)",
    )
    parser.add_argument("--dataset_name", type=str, default="SimpleStories/SimpleStories")
    parser.add_argument("--streaming", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_ctx", type=int, default=512)
    parser.add_argument(
        "--tokenizer_file_path",
        type=str,
        default="simple_stories_train/tokenizer/simplestories-tokenizer.json",
    )
    args = parser.parse_args()

    assert args.wandb_path.startswith("wandb:"), "Currently only supports wandb paths"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_info = RunInfo.from_path(args.wandb_path)
    model = GPT2Simple(GPT2SimpleConfig(**run_info.model_config_dict))
    model.to(device)
    model.eval()

    # Data loader
    dcfg = DatasetConfig(
        name=args.dataset_name,
        is_tokenized=False,
        tokenizer_file_path=args.tokenizer_file_path,
        streaming=bool(args.streaming),
        split=args.split,
        n_ctx=args.n_ctx,
        column_name="story",
    )
    loader, _ = create_data_loader(
        dcfg, batch_size=args.batch_size, buffer_size=1000, global_seed=0
    )
    it = iter(loader)

    # Where to hook: inputs to ln_1, ln_2 in each block, and ln_f input
    # We'll hook pre-forward to capture input tensor
    ln_paths: list[tuple[str, Any]] = []
    for i, block in enumerate(model.h):
        ln_paths.append((f"h.{i}.ln_1", block.ln_1))
        ln_paths.append((f"h.{i}.ln_2", block.ln_2))
    ln_paths.append(("ln_f", model.ln_f))

    running: dict[str, float] = {p: 0.0 for p, _ in ln_paths}
    counts: dict[str, int] = {p: 0 for p, _ in ln_paths}

    def make_hook(path: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_mod: nn.Module, inp: tuple[Any, ...], _out: Any | None = None) -> None:
            x: Float[Tensor, "batch seq hidden"] = inp[0]
            mu = x.mean(dim=-1, keepdim=True)
            sigma = (x - mu).pow(2).mean(dim=-1).add(args.eps).sqrt()
            val = float(sigma.mean().item())
            running[path] += val
            counts[path] += 1
            return None

        return hook

    handles = [m.register_forward_pre_hook(make_hook(p)) for p, m in ln_paths]

    # Iterate
    for _ in range(args.batches):
        try:
            batch = next(it)["input_ids"].to(torch.int)
        except StopIteration:
            break
        x = batch.to(device)
        x_in = x[:, :-1]
        _ = model(x_in, return_logits=False)

    for h in handles:
        h.remove()

    # Averages
    stats = {}
    for path in running:
        if counts[path] > 0:
            stats[path] = running[path] / counts[path]

    out = {"model_path": args.wandb_path, "stats": stats}
    out_path = REPO_ROOT / "out"
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / "ln-stats.json"
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)

    api = wandb.Api()
    run = api.run(args.wandb_path.removeprefix("wandb:"))
    run.upload_file(filename, root=out_path)

    print(f"Saved LN stats to {filename}")


if __name__ == "__main__":
    main()
