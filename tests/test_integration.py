"""
End-to-end integration tests — full pipeline without a real HuggingFace model.

Covers: policy detection → privilege allocation → rank-restricted generation
        → save/load round-trip → evaluate_nlpn → calibrate_privilege.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src
from src.policy import Policy, PolicyCompiler, PolicyAllocator
from src.train import build_deny_examples, evaluate_nlpn, calibrate_privilege, _DEFAULT_ALLOW
from src.enforcer import wrap_with_nlpn, set_privilege, detect_rmax, save_nlpn, load_nlpn


# ── Minimal fake language model ───────────────────────────────────────────────

class _TinyLM(nn.Module):
    """Smallest model compatible with the full NLPN pipeline."""
    VOCAB = 50

    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(self.VOCAB, 16)
        self.down_proj = nn.Linear(16, 16)   # targeted by _DEFAULT_TARGET_MODULES
        self.up_proj   = nn.Linear(16, 16)   # targeted by _DEFAULT_TARGET_MODULES
        self.q_proj    = nn.Linear(16, 16)   # attention — targeted by _DEFAULT_TARGET_MODULES
        self.lm_head   = nn.Linear(16, self.VOCAB)

    def forward(self, input_ids, **_):
        x = torch.relu(self.embed(input_ids))
        x = torch.relu(self.down_proj(x))
        x = self.up_proj(x)
        x = self.q_proj(x)
        return SimpleNamespace(logits=self.lm_head(x))

    def generate(self, input_ids, max_new_tokens=5, pad_token_id=0, do_sample=False, **_):
        tokens = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(tokens).logits[:, -1]
            next_tok = logits.argmax(-1).unsqueeze(-1)
            tokens = torch.cat([tokens, next_tok], dim=-1)
            if next_tok.item() in (0, 1):
                break
        return tokens


class _FakeTok:
    eos_token    = "</s>"
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True, padding=False, **_):
        ids = [max(2, abs(hash(w)) % 48) for w in text.split()[:6]] or [2]
        if return_tensors == "pt":
            t = torch.tensor([ids])
            if padding:
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}
            return {"input_ids": t}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(str(i) for i in seq if i > 1)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def wrapped_model():
    model = _TinyLM()
    wrap_with_nlpn(model, rmax=8, target_modules=["down_proj", "up_proj", "q_proj"])
    model.eval()
    return model

@pytest.fixture
def tokenizer():
    return _FakeTok()

@pytest.fixture
def policy():
    return Policy.from_text("DENY: salary\n  match: salary, wage\nDENY: medical\n  match: medical\n")


# ── Policy → privilege pipeline ───────────────────────────────────────────────

def test_policy_deny_sets_low_privilege(wrapped_model, tokenizer, policy):
    compiler  = PolicyCompiler(policy)
    allocator = PolicyAllocator(compiler, tokenizer, low_privilege=1)
    enc = tokenizer("What is the salary?", return_tensors="pt")
    _, g = allocator.generate(wrapped_model, enc["input_ids"], rmax=8, max_new_tokens=3)
    assert g == 1

def test_policy_allow_sets_full_privilege(wrapped_model, tokenizer, policy):
    compiler  = PolicyCompiler(policy)
    allocator = PolicyAllocator(compiler, tokenizer, low_privilege=1)
    enc = tokenizer("What are office hours?", return_tensors="pt")
    _, g = allocator.generate(wrapped_model, enc["input_ids"], rmax=8, max_new_tokens=3)
    assert g == 8

def test_output_differs_at_different_privilege(wrapped_model, tokenizer):
    enc = tokenizer("hello world", return_tensors="pt")
    set_privilege(wrapped_model, 1)
    out_low  = wrapped_model.generate(enc["input_ids"], max_new_tokens=4)
    set_privilege(wrapped_model, 8)
    out_high = wrapped_model.generate(enc["input_ids"], max_new_tokens=4)
    assert not torch.equal(out_low, out_high)

def test_attention_and_mlp_both_wrapped(wrapped_model):
    from src.nlpn import NLPNLinear
    wrapped_names = {
        name for name, m in wrapped_model.named_modules()
        if isinstance(m, NLPNLinear)
    }
    assert "down_proj" in wrapped_names   # MLP
    assert "q_proj"    in wrapped_names   # attention


# ── Save / load round-trip ────────────────────────────────────────────────────

def test_save_load_preserves_output(wrapped_model, tokenizer, tmp_path):
    enc = tokenizer("hello", return_tensors="pt")
    set_privilege(wrapped_model, 8)
    out_before = wrapped_model.generate(enc["input_ids"], max_new_tokens=3).clone()

    save_nlpn(wrapped_model, tmp_path / "ckpt", model_id="test/tiny")

    model2 = _TinyLM()
    wrap_with_nlpn(model2, rmax=8, target_modules=["down_proj", "up_proj", "q_proj"])
    model2.eval()
    load_nlpn(model2, tmp_path / "ckpt")

    set_privilege(model2, 8)
    out_after = model2.generate(enc["input_ids"], max_new_tokens=3)
    assert torch.equal(out_before, out_after)


# ── evaluate_nlpn ─────────────────────────────────────────────────────────────

def test_evaluate_nlpn_returns_expected_keys(wrapped_model, tokenizer, policy):
    deny_ex  = build_deny_examples(policy)
    result   = evaluate_nlpn(wrapped_model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=8, low_g=1)
    assert "deny_suppression_rate"   in result
    assert "allow_preservation_rate" in result
    assert result["low_g"] == 1
    assert result["rmax"]  == 8

def test_evaluate_nlpn_rates_in_range(wrapped_model, tokenizer, policy):
    deny_ex = build_deny_examples(policy)
    result  = evaluate_nlpn(wrapped_model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=8, low_g=1)
    assert 0.0 <= result["deny_suppression_rate"]   <= 1.0
    assert 0.0 <= result["allow_preservation_rate"] <= 1.0

def test_evaluate_nlpn_restores_full_privilege(wrapped_model, tokenizer, policy):
    from src.nlpn import NLPNLinear
    deny_ex = build_deny_examples(policy)
    evaluate_nlpn(wrapped_model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=8, low_g=1)
    for m in wrapped_model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 8


# ── calibrate_privilege ───────────────────────────────────────────────────────

def test_calibrate_returns_value_in_range(wrapped_model, tokenizer, policy):
    deny_ex = build_deny_examples(policy)
    low_g   = calibrate_privilege(wrapped_model, tokenizer, deny_ex, rmax=8)
    assert 1 <= low_g <= 8

def test_calibrate_restores_full_privilege(wrapped_model, tokenizer, policy):
    from src.nlpn import NLPNLinear
    deny_ex = build_deny_examples(policy)
    calibrate_privilege(wrapped_model, tokenizer, deny_ex, rmax=8)
    for m in wrapped_model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 8
