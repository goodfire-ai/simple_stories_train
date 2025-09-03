"""Utilities for ablating layernorm as in https://arxiv.org/abs/2507.02559."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from simple_stories_train.models.gpt2_simple import LayerNorm


def _iter_module_paths(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Module]]:
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        yield path, child
        yield from _iter_module_paths(child, path)


def enumerate_gpt2_simple_ln_paths(model: nn.Module) -> dict[str, list[str]]:
    """Return GPT-2 LN paths grouped by stage order: ln2_list (MLP), ln1_list (attn), ln_f.

    Assumes our custom GPT-2 structure with blocks in `h` and final `ln_f`.
    """
    ln1_list: list[str] = []
    ln2_list: list[str] = []
    ln_f_path: list[str] = []

    # Find blocks under h: sequential order matters
    if hasattr(model, "h") and isinstance(model.h, nn.ModuleList):  # type: ignore[attr-defined]
        for i, block in enumerate(model.h):  # type: ignore[attr-defined]
            if hasattr(block, "ln_1") and isinstance(block.ln_1, LayerNorm):
                ln1_list.append(f"h.{i}.ln_1")
            if hasattr(block, "ln_2") and isinstance(block.ln_2, LayerNorm):
                ln2_list.append(f"h.{i}.ln_2")

    # Final LN
    if hasattr(model, "ln_f") and isinstance(model.ln_f, LayerNorm):  # type: ignore[attr-defined]
        ln_f_path = ["ln_f"]

    return {"ln2": ln2_list, "ln1": ln1_list, "ln_f": ln_f_path}


def build_paper_order_paths(model: nn.Module) -> list[str]:
    groups = enumerate_gpt2_simple_ln_paths(model)
    # Stage 1: all ln_2 (MLP), Stage 2: all ln_1 (attn), Stage 3: ln_f
    order: list[str] = []
    order.extend(groups["ln2"])  # 0..L-1
    order.extend(groups["ln1"])  # 0..L-1
    order.extend(groups["ln_f"])  # final
    return order


def _resolve_module(parent: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    """Return (parent_module, child_name, child_module) for the given dotted path."""
    parts = path.split(".")
    obj: nn.Module = parent
    for part in parts[:-1]:
        obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]  # type: ignore[index]
        if not isinstance(obj, nn.Module):
            raise AttributeError(f"Path segment {part} is not a module for {path}")
    child_name = parts[-1]
    child = getattr(obj, child_name) if not child_name.isdigit() else obj[int(child_name)]  # type: ignore[index]
    if not isinstance(child, nn.Module):
        raise AttributeError(f"Leaf {child_name} is not a module for {path}")
    return obj, child_name, child


def set_ln_std_at_path(model: nn.Module, module_path: str, std_value: float) -> None:
    child = _resolve_module(model, module_path)[-1]
    assert isinstance(child, LayerNorm), (
        f"Module at {module_path} is not a custom LayerNorm: {type(child)}"
    )
    assert hasattr(child, "std"), (
        f"Custom LayerNorm at {module_path} does not expose a `std` attribute: {type(child)}"
    )
    child.std = std_value


@dataclass
class LnStats:
    model_id: str
    n_layer: int
    n_embd: int
    eps: float
    stats: dict[str, dict[str, float]]  # {module_path: {"sigma_avg": float}}


def load_ln_stats(path: Path) -> LnStats:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return LnStats(
        model_id=data["model_id"],
        n_layer=data["n_layer"],
        n_embd=data["n_embd"],
        eps=data["eps"],
        stats=data["stats"],
    )


def get_sigma_for_path(stats: LnStats, module_path: str) -> float:
    entry = stats.stats[module_path]
    if "sigma_avg" not in entry:
        raise KeyError(f"No sigma_avg for path {module_path} in stats file")
    return float(entry["sigma_avg"])
