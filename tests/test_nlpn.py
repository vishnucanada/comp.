import pytest
import torch
import torch.nn as nn

from src.nlpn import NLPNLinear


def _make_linear(din=8, dout=16) -> nn.Linear:
    layer = nn.Linear(din, dout)
    nn.init.normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


# ── Construction ──────────────────────────────────────────────────────────────


def test_shapes():
    layer = NLPNLinear(8, 16, rmax=4)
    assert layer.A.shape == (4, 8)
    assert layer.B.shape == (16, 4)
    assert layer.bias.shape == (16,)


def test_no_bias():
    layer = NLPNLinear(8, 16, rmax=4, bias=False)
    assert layer.bias is None


def test_default_privilege_is_rmax():
    layer = NLPNLinear(8, 16, rmax=4)
    assert layer.privilege == 4


# ── from_linear (SVD init) ────────────────────────────────────────────────────


def test_from_linear_shapes():
    linear = _make_linear(din=8, dout=16)
    layer = NLPNLinear.from_linear(linear, rmax=8)
    assert layer.A.shape == (8, 8)
    assert layer.B.shape == (16, 8)


def test_from_linear_bias_copied():
    linear = _make_linear()
    linear.bias.data.fill_(3.0)
    layer = NLPNLinear.from_linear(linear, rmax=4)
    assert torch.allclose(layer.bias.data, linear.bias.data)


def test_from_linear_no_bias():
    linear = nn.Linear(8, 16, bias=False)
    layer = NLPNLinear.from_linear(linear, rmax=4)
    assert layer.bias is None


def test_from_linear_reconstruction_at_rmax():
    linear = _make_linear(din=8, dout=8)
    rmax = 8
    layer = NLPNLinear.from_linear(linear, rmax=rmax)
    layer.privilege = rmax
    x = torch.randn(1, 8)
    out_orig = linear(x)
    out_nlpn = layer(x)
    assert torch.allclose(out_orig, out_nlpn, atol=1e-4), (
        f"Max diff: {(out_orig - out_nlpn).abs().max().item()}"
    )


# ── Forward & privilege ───────────────────────────────────────────────────────


def test_forward_output_shape():
    layer = NLPNLinear(8, 16, rmax=4)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 16)


def test_lower_privilege_changes_output():
    layer = NLPNLinear.from_linear(_make_linear(din=8, dout=8), rmax=8)
    x = torch.randn(1, 8)
    layer.privilege = 8
    out_full = layer(x).detach().clone()
    layer.privilege = 1
    out_low = layer(x).detach().clone()
    assert not torch.allclose(out_full, out_low)


def test_privilege_setter_valid():
    layer = NLPNLinear(8, 16, rmax=4)
    layer.privilege = 2
    assert layer.privilege == 2


def test_privilege_setter_invalid_zero():
    layer = NLPNLinear(8, 16, rmax=4)
    with pytest.raises(ValueError):
        layer.privilege = 0


def test_privilege_setter_invalid_over_rmax():
    layer = NLPNLinear(8, 16, rmax=4)
    with pytest.raises(ValueError):
        layer.privilege = 5


def test_extra_repr():
    layer = NLPNLinear(8, 16, rmax=4)
    r = layer.extra_repr()
    assert "rmax=4" in r
    assert "privilege=4" in r


# ── Nested structure property ─────────────────────────────────────────────────


def test_nested_structure():
    """rank(W(g)) <= rank(W(g+1)) — higher privilege can only add information."""
    layer = NLPNLinear.from_linear(_make_linear(din=16, dout=16), rmax=8)
    ranks = []
    for g in range(1, 9):
        layer.privilege = g
        W = layer.B[:, :g] @ layer.A[:g, :]
        rank = torch.linalg.matrix_rank(W).item()
        ranks.append(rank)
    # ranks must be non-decreasing
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))


# ── W(g) eval cache ───────────────────────────────────────────────────────────


def test_cache_populated_in_eval():
    layer = NLPNLinear.from_linear(_make_linear(), rmax=4)
    layer.eval()
    x = torch.randn(1, 8)
    layer(x)
    assert layer._W_cache is not None
    assert layer._W_cache_g == layer.privilege


def test_cache_not_populated_in_train():
    layer = NLPNLinear.from_linear(_make_linear(), rmax=4)
    layer.train()
    x = torch.randn(1, 8)
    layer(x)
    assert layer._W_cache is None


def test_cache_invalidated_on_privilege_change():
    layer = NLPNLinear.from_linear(_make_linear(), rmax=4)
    layer.eval()
    x = torch.randn(1, 8)
    layer(x)
    assert layer._W_cache is not None
    layer.privilege = 2
    assert layer._W_cache is None


def test_cache_gives_same_output():
    layer = NLPNLinear.from_linear(_make_linear(), rmax=4)
    layer.eval()
    x = torch.randn(3, 8)
    out1 = layer(x).detach().clone()
    out2 = layer(x).detach().clone()  # should use cache
    assert torch.allclose(out1, out2)


def test_cache_recomputes_after_privilege_change():
    layer = NLPNLinear.from_linear(_make_linear(din=8, dout=8), rmax=8)
    layer.eval()
    x = torch.randn(1, 8)
    layer.privilege = 8
    out_full = layer(x).detach().clone()
    layer.privilege = 1
    out_low = layer(x).detach().clone()
    assert not torch.allclose(out_full, out_low)
