import json
import pytest
import torch
import torch.nn as nn
from src.nlpn import NLPNLinear
from src.enforcer import wrap_with_nlpn, set_privilege, get_rmax, detect_rmax, save_nlpn, load_nlpn


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.other = nn.Linear(8, 4)  # should NOT be wrapped


class _TinyTransformer(nn.Module):
    """Minimal model with both MLP and attention-style projections."""
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(8, 8)
        self.v_proj  = nn.Linear(8, 8)
        self.o_proj  = nn.Linear(8, 8)
        self.fc1     = nn.Linear(8, 16)
        self.fc2     = nn.Linear(16, 8)


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
            self.layer_a = nn.Linear(8, 16)
            self.layer_b = nn.Linear(16, 8)
    with pytest.raises(RuntimeError):
        wrap_with_nlpn(_CustomModel(), rmax=4)

def test_wrap_attention_projections():
    model = _TinyTransformer()
    wrap_with_nlpn(model, rmax=4, target_modules=["q_proj", "v_proj", "o_proj", "fc1", "fc2"])
    assert isinstance(model.q_proj, NLPNLinear)
    assert isinstance(model.v_proj, NLPNLinear)
    assert isinstance(model.o_proj, NLPNLinear)
    assert isinstance(model.fc1, NLPNLinear)
    assert isinstance(model.fc2, NLPNLinear)

def test_wrap_mlp_and_attention_same_rmax():
    model = _TinyTransformer()
    wrap_with_nlpn(model, rmax=4, target_modules=["q_proj", "v_proj", "fc1", "fc2"])
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert m.rmax == 4


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


# ── save_nlpn / load_nlpn ─────────────────────────────────────────────────────

def _wrapped(rmax=4, targets=("fc1", "fc2")):
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=rmax, target_modules=list(targets))
    return model


def test_save_creates_files(tmp_path):
    model = _wrapped()
    save_nlpn(model, tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "nlpn_weights.pt").exists()
    assert (tmp_path / "ckpt" / "nlpn_config.json").exists()

def test_save_config_records_model_id(tmp_path):
    model = _wrapped()
    save_nlpn(model, tmp_path / "ckpt", model_id="Qwen/Qwen2.5-0.5B")
    cfg = json.loads((tmp_path / "ckpt" / "nlpn_config.json").read_text())
    assert cfg["model_id"] == "Qwen/Qwen2.5-0.5B"

def test_save_config_layer_count(tmp_path):
    model = _wrapped()
    save_nlpn(model, tmp_path / "ckpt")
    cfg = json.loads((tmp_path / "ckpt" / "nlpn_config.json").read_text())
    assert cfg["n_layers"] == 2  # fc1 and fc2

def test_save_no_nlpn_raises():
    model = _TinyModel()  # not wrapped
    with pytest.raises(ValueError, match="No NLPNLinear"):
        save_nlpn(model, "/tmp/should_not_exist")

def test_load_restores_weights(tmp_path):
    model = _wrapped()
    # stamp B matrices with a known value
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            nn.init.ones_(m.B)

    save_nlpn(model, tmp_path / "ckpt")

    model2 = _wrapped()
    load_nlpn(model2, tmp_path / "ckpt")

    for m1, m2 in zip(
        [m for m in model.modules()  if isinstance(m, NLPNLinear)],
        [m for m in model2.modules() if isinstance(m, NLPNLinear)],
    ):
        assert torch.allclose(m1.A.data, m2.A.data)
        assert torch.allclose(m1.B.data, m2.B.data)

def test_load_returns_model(tmp_path):
    model = _wrapped()
    save_nlpn(model, tmp_path / "ckpt")
    model2 = _wrapped()
    result = load_nlpn(model2, tmp_path / "ckpt")
    assert result is model2

def test_load_no_match_raises(tmp_path):
    model = _wrapped(targets=("fc1",))
    save_nlpn(model, tmp_path / "ckpt")
    # Different target layer — no name overlap with saved config
    model2 = _wrapped(targets=("fc2",))
    with pytest.raises(ValueError, match="No NLPNLinear layers matched"):
        load_nlpn(model2, tmp_path / "ckpt")

def test_save_load_preserves_inference(tmp_path):
    model = _wrapped()
    x = torch.randn(1, 8)
    set_privilege(model, 4)
    out_before = model.fc1(x).detach().clone()

    save_nlpn(model, tmp_path / "ckpt")

    model2 = _wrapped()
    load_nlpn(model2, tmp_path / "ckpt")
    set_privilege(model2, 4)
    out_after = model2.fc1(x)

    assert torch.allclose(out_before, out_after, atol=1e-6)
