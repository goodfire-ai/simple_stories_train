"""
Tests for GPT-2 conversion between custom implementation and HuggingFace.
"""

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel

from simple_stories_train.models.gpt2 import (
    GPT2,
    GPT2Config,
    convert_gpt2_to_hf_gpt2,
    convert_hf_gpt2_to_gpt2,
)


@torch.inference_mode()
def test_convert_gpt2_to_hf_gpt2() -> None:
    """Validate custom -> HF conversion produces identical logits."""
    # Small config for speed
    config = GPT2Config(
        model_type="GPT2", block_size=64, vocab_size=50257, n_layer=2, n_head=2, n_embd=128
    )
    custom_model = GPT2(config)
    custom_model.eval()

    hf_model = convert_gpt2_to_hf_gpt2(custom_model)
    hf_model.eval()

    # Random input ids within vocab range
    batch_size = 2
    seq_len = 16
    inputs: Int[Tensor, "batch pos"] = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    hf_logits = hf_model(input_ids=inputs).logits  # type: ignore[arg-type]
    custom_logits, _ = custom_model(inputs)

    assert custom_logits is not None
    torch.testing.assert_close(hf_logits, custom_logits, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_convert_hf_gpt2_to_gpt2() -> None:
    """Validate HF -> custom conversion produces identical logits."""
    # Construct a tiny HF GPT-2 config/model to avoid network downloads

    hf_config = HFGPT2Config(
        vocab_size=50257,
        n_positions=64,
        n_ctx=64,
        n_layer=2,
        n_head=2,
        n_embd=128,
        activation_function="gelu_new",
        tie_word_embeddings=True,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    hf_model.eval()

    custom_model = convert_hf_gpt2_to_gpt2(hf_model)
    custom_model.eval()

    batch_size = 2
    seq_len = 16
    inputs: Int[Tensor, "batch pos"] = torch.randint(
        0, hf_model.config.vocab_size, (batch_size, seq_len)
    )

    hf_logits = hf_model(input_ids=inputs).logits  # type: ignore[arg-type]
    custom_logits, _ = custom_model(inputs)

    assert custom_logits is not None
    torch.testing.assert_close(hf_logits, custom_logits, rtol=1e-5, atol=1e-5)
