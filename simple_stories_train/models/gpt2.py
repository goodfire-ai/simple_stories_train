import inspect
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel

from simple_stories_train.run_info import RunInfo
from simple_stories_train.utils import print0

# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false


class GPT2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash_attention: bool = True


class NewGELU(nn.Module):
    def forward(self, input: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        # Use pre-stored stds instead of computing them on the fly
        self.std: float | None = None

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        if self.std is None:
            residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        else:
            residual_std = self.std

        residual = (residual - residual_mean) / residual_std
        return residual * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    bias: Tensor

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = True  # type: ignore
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        # calculate q, k, v for all heads in batch
        # move head dimension forward to be the batch dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash_attention:
            # use PyTorch SDPA
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = True  # type: ignore

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe: nn.Embedding = nn.Embedding(config.block_size, config.n_embd)
        self._h: list[Block] = [Block(config) for _ in range(config.n_layer)]
        self.h: nn.ModuleList = nn.ModuleList(self._h)
        self.ln_f: LayerNorm = LayerNorm(config.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = True  # type: ignore
        self.wte.weight = self.lm_head.weight  # type: ignore[reportAttributeAccessIssue]

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
            )
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)  # type: ignore[arg-type]
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(
        self,
        idx: Int[Tensor, "batch pos"],
        targets: Int[Tensor, "batch pos"] | None = None,
        return_logits: bool = True,
    ) -> tuple[
        Float[Tensor, "batch pos vocab"] | None,
        Float[Tensor, ""] | None,
    ]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits: Tensor = self.lm_head(x)
        loss: Tensor | None
        if targets is not None:
            if targets.dtype != torch.long:
                targets = targets.to(torch.long)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            loss = None

        out_logits: Tensor | None = logits
        if not return_logits:
            out_logits = None

        return out_logits, loss

    @classmethod
    def from_run_info(cls, run_info: RunInfo) -> "GPT2":
        """Create a GPT-2 model from a RunInfo, loading weights from its checkpoint."""
        model = cls(GPT2Config(**run_info.model_config_dict))
        state_dict = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "GPT2":
        """Create a GPT-2 model from a wandb string or a local path.

        Args:
            model_path:
                - W&B strings: 'wandb:goodfire/spd-play/runs/152i5k4r'
                - Local: path to a .pt checkpoint file
        """
        run_info = RunInfo.from_path(model_path)
        return cls.from_run_info(run_info)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        zero_stage: int,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optim_group = optim_groups[0]
            optimizer: torch.optim.Optimizer = ZeroRedundancyOptimizer(  # type: ignore[assignment]
                **optim_group,  # type: ignore[arg-type]
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])  # type: ignore[arg-type]
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Float[Tensor, "... pos"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Float[Tensor, "... pos"]:
        # Keep track of whether input was 1D and ensure input has batch dimension
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)

        batch_size = idx.size(0)
        not_completed = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if not not_completed.any():
                break

            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            assert logits is not None
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
            else:
                probs = torch.zeros_like(logits)
                probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                not_completed = not_completed & (idx_next[:, -1] != eos_token_id)
                update_mask = not_completed.unsqueeze(-1)
                idx_next = torch.where(
                    update_mask, idx_next, torch.full_like(idx_next, eos_token_id)
                )

            idx = torch.cat((idx, idx_next), dim=1)

        if is_1d:
            idx = idx.squeeze(0)

        return idx


def _build_mapping(
    direction: Literal["custom_to_hf", "hf_to_custom"], n_layer: int
) -> list[tuple[str, str, bool]]:
    base_pairs: list[tuple[str, str, bool]] = [
        ("wte.weight", "transformer.wte.weight", False),
        ("wpe.weight", "transformer.wpe.weight", False),
        ("ln_f.weight", "transformer.ln_f.weight", False),
        ("ln_f.bias", "transformer.ln_f.bias", False),
        ("lm_head.weight", "lm_head.weight", False),
    ]

    layer_pairs: list[tuple[str, str, bool]] = []
    for i in range(n_layer):
        c_prefix = f"h.{i}."
        h_prefix = f"transformer.h.{i}."
        layer_pairs.extend(
            [
                (f"{c_prefix}ln_1.weight", f"{h_prefix}ln_1.weight", False),
                (f"{c_prefix}ln_1.bias", f"{h_prefix}ln_1.bias", False),
                (f"{c_prefix}ln_2.weight", f"{h_prefix}ln_2.weight", False),
                (f"{c_prefix}ln_2.bias", f"{h_prefix}ln_2.bias", False),
                (f"{c_prefix}attn.c_attn.weight", f"{h_prefix}attn.c_attn.weight", True),
                (f"{c_prefix}attn.c_attn.bias", f"{h_prefix}attn.c_attn.bias", False),
                (f"{c_prefix}attn.c_proj.weight", f"{h_prefix}attn.c_proj.weight", True),
                (f"{c_prefix}attn.c_proj.bias", f"{h_prefix}attn.c_proj.bias", False),
                (f"{c_prefix}mlp.c_fc.weight", f"{h_prefix}mlp.c_fc.weight", True),
                (f"{c_prefix}mlp.c_fc.bias", f"{h_prefix}mlp.c_fc.bias", False),
                (f"{c_prefix}mlp.c_proj.weight", f"{h_prefix}mlp.c_proj.weight", True),
                (f"{c_prefix}mlp.c_proj.bias", f"{h_prefix}mlp.c_proj.bias", False),
            ]
        )

    mapping = base_pairs + layer_pairs
    if direction == "custom_to_hf":
        return mapping
    return [(dst, src, transpose) for (src, dst, transpose) in mapping]


def _resolve_tensor(module: nn.Module, path: str) -> Tensor:
    """Get tensor from module by path.

    E.g. _resolve_tensor(module, "transformer.h.0.attn.c_attn.weight")
    will return the weight tensor for the first attention layer.
    """

    obj: Any = module
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    assert isinstance(obj, Tensor)
    return obj


@torch.inference_mode()
def _copy_by_mapping(src: nn.Module, dst: nn.Module, mapping: list[tuple[str, str, bool]]) -> None:
    for src_path, dst_path, transpose in mapping:
        src_tensor = _resolve_tensor(src, src_path)
        dst_tensor = _resolve_tensor(dst, dst_path)
        tensor_to_copy = src_tensor.t().contiguous() if transpose else src_tensor
        dst_tensor.copy_(tensor_to_copy)


def convert_hf_gpt2_to_gpt2(hf_model: GPT2LMHeadModel) -> GPT2:
    """Convert a HuggingFace GPT2LMHeadModel to our custom GPT2.

    Args:
        hf_model: HuggingFace GPT2LMHeadModel instance

    Returns:
        Our custom GPT2 model with weights copied from the HF model
    """
    custom_config = GPT2Config(
        block_size=hf_model.config.n_ctx,
        vocab_size=hf_model.config.vocab_size,
        n_layer=hf_model.config.n_layer,
        n_head=hf_model.config.n_head,
        n_embd=hf_model.config.n_embd,
        flash_attention=True,
    )
    custom_model = GPT2(custom_config)
    mapping = _build_mapping("hf_to_custom", custom_model.config.n_layer)
    _copy_by_mapping(src=hf_model, dst=custom_model, mapping=mapping)
    return custom_model


def convert_gpt2_to_hf_gpt2(custom_model: GPT2) -> GPT2LMHeadModel:
    """Convert custom GPT-2 model to HuggingFace GPT2LMHeadModel.

    Args:
        custom_model: The custom GPT-2 model to convert

    Returns:
        The converted HuggingFace GPT2LMHeadModel
    """
    hf_config = HFGPT2Config(
        vocab_size=custom_model.config.vocab_size,
        n_positions=custom_model.config.block_size,
        n_ctx=custom_model.config.block_size,
        n_layer=custom_model.config.n_layer,
        n_head=custom_model.config.n_head,
        n_embd=custom_model.config.n_embd,
        activation_function="gelu_new",
        n_inner=None,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=True,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    mapping = _build_mapping("custom_to_hf", custom_model.config.n_layer)
    _copy_by_mapping(src=custom_model, dst=hf_model, mapping=mapping)
    hf_model.eval()
    return hf_model
