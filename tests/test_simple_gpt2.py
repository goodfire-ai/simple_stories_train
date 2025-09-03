import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from simple_stories_train.models.gpt2_simple import LayerNorm as CustomLayerNorm


def _make_modules(d_model: int, eps: float) -> tuple[CustomLayerNorm, torch.nn.LayerNorm]:
    ours = CustomLayerNorm(n_embd=d_model, eps=eps)
    theirs = torch.nn.LayerNorm(normalized_shape=d_model, eps=eps, elementwise_affine=True)
    with torch.no_grad():
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)
    return ours, theirs


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 1, 8), (4, 5, 64)])
@pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-3])
def test_layernorm_matches_torch(shape: tuple[int, int, int], eps: float) -> None:
    batch, pos, d_model = shape
    ours, theirs = _make_modules(d_model=d_model, eps=eps)
    x: Float[Tensor, "batch pos d_model"] = torch.randn(batch, pos, d_model, dtype=torch.float32)
    out_ours: Float[Tensor, "batch pos d_model"] = ours(x)
    out_theirs: Float[Tensor, "batch pos d_model"] = theirs(x)
    torch.testing.assert_close(out_ours, out_theirs, rtol=1e-5, atol=1e-6)


def test_layernorm_matches_with_random_affine() -> None:
    d_model: int = 128
    eps: float = 1e-5
    ours, theirs = _make_modules(d_model=d_model, eps=eps)
    torch.manual_seed(0)
    weight: Float[Tensor, " d_model"] = torch.randn(d_model, dtype=torch.float32)
    bias: Float[Tensor, " d_model"] = torch.randn(d_model, dtype=torch.float32)
    with torch.no_grad():
        ours.weight.copy_(weight)
        ours.bias.copy_(bias)
        theirs.weight.copy_(weight)
        theirs.bias.copy_(bias)
    x: Float[Tensor, "batch pos d_model"] = torch.randn(3, 7, d_model, dtype=torch.float32)
    out_ours: Float[Tensor, "batch pos d_model"] = ours(x)
    out_theirs: Float[Tensor, "batch pos d_model"] = theirs(x)
    torch.testing.assert_close(out_ours, out_theirs, rtol=1e-5, atol=1e-6)
