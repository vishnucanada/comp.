import pytest
import torch
import torch.nn as nn
from src.nlpn import NLPNLinear
from src.enforcer import wrap_with_nlpn, set_privilege, get_rmax, detect_rmax


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.other = nn.Linear(8, 4)  # should NOT be wrapped


# ── wrap_with_nlpn ────────────────────────────────────────────────────────────

def test_wrap_replaces_target_layers():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=4, target_modules=["fc1", "fc2"])
    assert isinstance(model.fc1, NLPNLinear)
    assert isinstance(model.fc2, NLPNLinear)

def test_wrap_skips_non_target_layers():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=4, target_modules=["fc1", "fc2"])
    assert isinstance(model.other, nn.Linear)
    assert not isinstance(model.other, NLPNLinear)

def test_wrap_returns_model():
    model = _TinyModel()
    returned = wrap_with_nlpn(model, rmax=4, target_modules=["fc1"])
    assert returned is model

def test_wrap_sets_rmax():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=6, target_modules=["fc1"])
    assert model.fc1.rmax == 6

def test_wrap_no_match_raises():
    model = _TinyModel()
    with pytest.raises(RuntimeError, match="No matching"):
        wrap_with_nlpn(model, rmax=4, target_modules=["nonexistent"])

def test_wrap_default_modules_raises_on_unknown_arch():
    class _CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_a = nn.Linear(8, 16)  # names not in _DEFAULT_TARGET_MODULES
            self.layer_b = nn.Linear(16, 8)
    with pytest.raises(RuntimeError):
        wrap_with_nlpn(_CustomModel(), rmax=4)


# ── set_privilege ─────────────────────────────────────────────────────────────

def test_set_privilege_updates_all_layers():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=8, target_modules=["fc1", "fc2"])
    set_privilege(model, 3)
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 3

def test_set_privilege_invalid_raises():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=8, target_modules=["fc1"])
    with pytest.raises(ValueError):
        set_privilege(model, 0)


# ── get_rmax ──────────────────────────────────────────────────────────────────

def test_get_rmax_returns_value():
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=5, target_modules=["fc1"])
    assert get_rmax(model) == 5

def test_get_rmax_no_nlpn_raises():
    model = _TinyModel()
    with pytest.raises(ValueError, match="No NLPNLinear"):
        get_rmax(model)


# ── detect_rmax ───────────────────────────────────────────────────────────────

def test_detect_rmax_returns_min_dim():
    model = _TinyModel()
    # fc1 is Linear(8, 16) → min(8, 16) = 8
    rmax = detect_rmax(model, target_modules=["fc1"])
    assert rmax == 8

def test_detect_rmax_no_match_raises():
    model = _TinyModel()
    with pytest.raises(RuntimeError, match="No target layers"):
        detect_rmax(model, target_modules=["nonexistent"])
