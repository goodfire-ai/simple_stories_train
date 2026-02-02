"""
Unified training script for multiple model families (Llama, GPT-2).

Usage:
```bash
python -m simple_stories_train.train [CONFIG.yaml] [--key1 value1 --key2 value2 ...]
```
- CONFIG.yaml contains the training config. If not provided, a default config is used.
- Override values with dotted notation (e.g., --train_dataset_config.name my_dataset).

To run on multiple GPUs:
```bash
torchrun --standalone --nproc_per_node=N -m simple_stories_train.train ...
```
where N is the number of GPUs.
"""

import math
import os
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import fire
import numpy as np
import torch
import torch._inductor.config as torch_inductor_config
import torch.distributed as dist
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from simple_stories_train.base_config import BaseConfig
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.ln_free import (
    build_paper_order_paths,
    set_ln_std_at_path,
)
from simple_stories_train.models import MODEL_CLASSES, ModelConfig
from simple_stories_train.run_info import RunInfo
from simple_stories_train.utils import (
    REPO_ROOT,
    is_checkpoint_step,
    load_config,
    log_generations,
    log_metrics,
    print0,
    save_configs,
    save_model,
)


class Config(BaseConfig):
    wandb_project: str | None = Field(
        None, description="WandB project name. If None, will not use WandB."
    )
    train_dataset_config: DatasetConfig = Field(
        DatasetConfig(
            name="SimpleStories/SimpleStories",
            is_tokenized=False,
            tokenizer_file_path="simple_stories_train/tokenizer/simplestories-tokenizer.json",
            streaming=True,
            split="train",
            n_ctx=1024,
            seed=0,
            column_name="story",
        ),
        description="Dataset config for training",
    )
    val_dataset_config: DatasetConfig = Field(
        DatasetConfig(
            name="SimpleStories/SimpleStories",
            is_tokenized=False,
            tokenizer_file_path="simple_stories_train/tokenizer/simplestories-tokenizer.json",
            streaming=True,
            split="test",
            n_ctx=1024,
            seed=0,
            column_name="story",
        ),
        description="Dataset config for validation",
    )
    output_dir: Path = Field(
        REPO_ROOT / "out", description="Directory to write logs and checkpoints"
    )
    # Model configuration (discriminated union)
    model: ModelConfig = Field(..., description="Model configuration")
    batch_size: PositiveInt = Field(4, description="Batch size")
    num_iterations: PositiveInt = Field(50, description="Number of training steps")
    inference_only: bool = Field(False, description="If True, don't update gradients")
    learning_rate: PositiveFloat = Field(1e-4, description="Learning rate")
    warmup_iters: NonNegativeInt = Field(
        0, description="Number of iterations to warmup the learning rate"
    )
    learning_rate_decay_frac: PositiveFloat = Field(
        1.0, ge=0, le=1, description="Fraction of lr to decay to. 0 decays to 0, 1 doesn't decay"
    )
    weight_decay: NonNegativeFloat = Field(0.1, description="Weight decay")
    grad_clip: NonNegativeFloat | None = Field(1.0, description="Maximum gradient magnitude")
    val_loss_every: NonNegativeInt = Field(
        0, description="Every how many steps to evaluate val loss?"
    )
    val_max_steps: NonNegativeInt = Field(
        20, description="Max number of batches to use for validation"
    )
    train_log_every: NonNegativeInt = Field(100, description="How often to log train loss?")
    sample_every: NonNegativeInt = Field(0, description="How often to sample from the model?")
    tensorcores: bool = Field(True, description="Use TensorCores?")
    device: str | None = Field(None, description="Device to use. If None, will autodetect.")
    compile: bool = Field(True, description="Compile the model?")
    flash_attention: bool = Field(True, description="Use FlashAttention?")
    dtype: Literal["float32", "float16", "bfloat16"] = Field("bfloat16", description="Data type")
    zero_stage: Literal[0, 1, 2, 3] = Field(
        0, description="Zero redundancy optimizer stage (0/1/2/3)"
    )
    intermediate_checkpoints: bool = Field(
        False, description="Save intermediate checkpoints (done at steps 0, 1, 2, 4, 8, ...)?"
    )
    from_pretrained: str | Path | None = Field(
        None, description="Path to a wandb string or a local path to a checkpoint to finetune from"
    )
    # LN ablation configs
    enable_ln_ablation: bool = Field(
        False, description="Enable staged setting of LayerNorm std (no module swapping)"
    )
    n_steps_between_ln_ablation: NonNegativeInt = Field(
        0, description="Steps between replacing successive LN layers"
    )
    ln_ablation_start_step: NonNegativeInt = Field(
        1, description="First global step eligible for LN replacement"
    )
    ln_ablation_max_to_remove: NonNegativeInt | None = Field(
        None, description="Optional cap on number of LN layers to replace"
    )


def maybe_replace_one_ln(
    raw_model: nn.Module,
    config: Config,
    step: int,
    ln_order_paths: list[str],
    ln_stds: dict[str, float],
    replaced_count: int,
) -> int:
    """Conditionally set a single LayerNorm's fixed std according to the schedule.

    Returns the updated `replaced_count`.
    """

    can_replace_more = (
        config.ln_ablation_max_to_remove is None
        or replaced_count < config.ln_ablation_max_to_remove
    )
    due = (
        step >= config.ln_ablation_start_step
        and (step - config.ln_ablation_start_step) % config.n_steps_between_ln_ablation == 0
    )
    if due and can_replace_more and replaced_count < len(ln_order_paths):
        path_to_replace = ln_order_paths[replaced_count]
        sigma_avg = ln_stds[path_to_replace]
        set_ln_std_at_path(raw_model, path_to_replace, std_value=sigma_avg)
        replaced_count += 1
        print0(f"Replace LN at step {step}: {path_to_replace} (std={sigma_avg:.6f})")
    return replaced_count


def main(config_path_or_obj: Path | str | Config | None = None, **kwargs: Any) -> None:
    print0(f"Running pytorch {torch.__version__}")
    load_dotenv(override=True)
    config = load_config(config_path_or_obj, config_model=Config, updates=kwargs)

    B = config.batch_size
    T = config.train_dataset_config.n_ctx

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        zero_stage = config.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        if config.device:
            device = config.device
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # dtype context
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    # TF32
    if config.tensorcores:
        torch.set_float32_matmul_precision("high")

    # Instantiate model using discriminated union config
    model_cls = MODEL_CLASSES[config.model.model_type]
    model: nn.Module = model_cls(config.model)

    # Load pretrained weights
    if config.from_pretrained is not None:
        assert hasattr(model_cls, "from_pretrained"), (
            f"Model {config.model.model_type} does not support from_pretrained"
        )
        pretrained_model = model_cls.from_pretrained(config.from_pretrained)  # pyright: ignore[reportAttributeAccessIssue]
        model.load_state_dict(pretrained_model.state_dict())
    model.to(device)

    model.train()
    model.to(device)
    if config.compile:
        if device_type == "cpu":
            warnings.warn(
                "compile may not be compatible with cpu, use `--compile=False` if issues",
                stacklevel=1,
            )
        if hasattr(torch_inductor_config, "coordinate_descent_tuning"):
            torch_inductor_config.coordinate_descent_tuning = True
        print0("compiling the model...")
        model = cast(nn.Module, torch.compile(model))  # type: ignore[reportArgumentType]

    train_loader, train_tokenizer = create_data_loader(
        dataset_config=config.train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    train_iter = iter(train_loader)

    val_loader, _ = create_data_loader(
        dataset_config=config.val_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        split_for_ddp=False,  # Don't split validation data - all ranks evaluate same data
    )

    # logging
    if config.wandb_project is not None and master_process:
        run = wandb.init(project=config.wandb_project, config=config.model_dump(mode="json"))
        run.name = f"{config.model.model_type}-{run.name}"

    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model: nn.Module = model.module if ddp else model  # type: ignore[attr-defined]

    # optimizer
    optimizer = raw_model.configure_optimizers(  # type: ignore[attr-defined]
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=zero_stage,
    )

    # lr schedule
    def get_lr(it: int) -> float:
        min_lr = config.learning_rate * config.learning_rate_decay_frac
        if it < config.warmup_iters:
            return config.learning_rate * (it + 1) / config.warmup_iters
        if it > config.num_iterations:
            return min_lr
        decay_ratio = (it - config.warmup_iters) / (config.num_iterations - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (config.learning_rate - min_lr)

    # For LN ablation
    ln_order_paths: list[str] = []
    ln_stds: dict[str, float] | None = None
    if config.enable_ln_ablation:
        assert isinstance(config.from_pretrained, str) and config.from_pretrained.startswith(
            "wandb:"
        ), "Currently only supports wandb paths"
        assert config.from_pretrained is not None
        ln_stds = RunInfo.from_path(config.from_pretrained).ln_stds
        assert ln_stds is not None, "enable_ln_ablation=True but ln_stds is None"
        ln_order_paths = build_paper_order_paths(raw_model)

    # IO dirs
    logfile = None
    checkpoints_dir = None
    output_dir = None
    if config.output_dir and master_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(config.output_dir) / f"{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logfile = output_dir / "main.log"
        with open(logfile, "w") as f:
            pass
        save_configs(
            output_dir,
            config_dict=config.model_dump(mode="json"),
            model_config_dict=config.model.model_dump(mode="json"),
            ln_stds=ln_stds,
        )
        # Save tokenizer to output_dir alongside configs and upload to W&B if enabled
        tokenizer_file = output_dir / "tokenizer.json"
        train_tokenizer.save(str(tokenizer_file))
        print0(f"Saved tokenizer to {tokenizer_file}")
        if config.wandb_project is not None and master_process:
            wandb.save(str(tokenizer_file), policy="now", base_path=output_dir)
            print0(f"Saved tokenizer to wandb from {str(tokenizer_file)}")
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if config.intermediate_checkpoints:
            save_model(checkpoints_dir, raw_model, step=0, wandb_project=config.wandb_project)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    generations: list[list[Any]] = []
    # For LN ablation
    replaced_count = 0

    for step in range(1, config.num_iterations + 1):
        last_step = step == config.num_iterations

        # validation
        if config.val_loss_every > 0 and (step % config.val_loss_every == 0 or last_step):
            model.eval()
            val_loader_iter = iter(val_loader)
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(config.val_max_steps):
                    try:
                        bat = next(val_loader_iter)["input_ids"].to(torch.long)
                    except StopIteration:
                        break
                    x = bat.view(B, T)[:, :-1]
                    y = bat.view(B, T)[:, 1:]
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += float(loss.item()) if loss is not None else 0.0
                val_loss /= config.val_max_steps
            if config.wandb_project is not None and master_process:
                log_metrics(step, {"val_loss": val_loss})
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")

        # sample generations
        if (
            config.sample_every > 0 and (step % config.sample_every == 0 or last_step)
        ) and master_process:
            model.eval()
            start_ids = [train_tokenizer.token_to_id("[EOS]")]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = cast(Any, raw_model).generate(
                xg, max_new_tokens, temperature=temperature, top_k=top_k
            )
            print0("---------------")
            print0(train_tokenizer.decode(yg[0].tolist()))
            print0("---------------")
            if config.wandb_project is not None and master_process:
                generations.append([step, train_tokenizer.decode(yg[0].tolist())])
                log_generations(step, generations)

        if last_step:
            break

        # training
        model.train()

        if config.enable_ln_ablation and config.n_steps_between_ln_ablation > 0:
            assert ln_stds is not None, "enable_ln_ablation=True but ln_stds is None"
            # maybe set one LN std according to schedule
            replaced_count = maybe_replace_one_ln(
                raw_model=raw_model,
                config=config,
                step=step,
                ln_order_paths=ln_order_paths,
                ln_stds=ln_stds,
                replaced_count=replaced_count,
            )
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        try:
            bat = next(train_iter)["input_ids"].to(torch.long)
        except StopIteration:
            print0("Depleted train_loader, resetting for next epoch")
            train_iter = iter(train_loader)
            bat = next(train_iter)["input_ids"].to(torch.long)

        x = bat.view(B, T)[:, :-1]
        y = bat.view(B, T)[:, 1:]
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y, return_logits=False)
        if not config.inference_only:
            loss.backward()  # type: ignore[arg-type]
        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # type: ignore[arg-type]
        lossf_value = float(loss.detach().item())  # type: ignore[union-attr]
        norm = None
        if config.grad_clip is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        if step % config.train_log_every == 0:
            tokens_per_second = ddp_world_size * B * T / (t1 - t0)
            norm_str = f"norm {norm:.4f}" if norm is not None else ""
            print0(
                f"step {step:4d}/{config.num_iterations} | loss {lossf_value:.6f} | {norm_str} | "
                f"lr {lr:.2e} | {(t1 - t0) * 1000:.2f}ms | {tokens_per_second:.0f} tok/s"
            )
        if config.wandb_project is not None and master_process:
            log_metrics(step, {"train_loss": lossf_value, "lr": lr})
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"step:{step} loss:{lossf_value}\n")

        if (
            checkpoints_dir is not None
            and master_process
            and (
                (config.intermediate_checkpoints and is_checkpoint_step(step))
                or step == config.num_iterations - 1
            )
        ):
            save_model(checkpoints_dir, raw_model, step=step, wandb_project=config.wandb_project)

        if step > 1 and (step > config.num_iterations - 20):
            timings.append(t1 - t0)

    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings) * 1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if ddp:
        destroy_process_group()

    if config.wandb_project is not None and master_process:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
