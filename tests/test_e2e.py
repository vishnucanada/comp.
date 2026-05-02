"""
End-to-end tests for the full comp. pipeline.

Uses a tiny in-process model so no network or GPU is required.
Covers: parse → wrap → compile → allocate → train → save/load → GDPR → evaluate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src
from src.enforcer import load_nlpn, save_nlpn, set_privilege, wrap_with_nlpn
from src.gdpr import AuditLog, GDPRAllocator, GDPRPolicyParser, verify_audit_log
from src.policy import Policy, PolicyAllocator, PolicyCompiler
from src.train import (
    _DEFAULT_ALLOW,
    TrainConfig,
    build_deny_examples,
    calibrate_privilege,
    evaluate_nlpn,
)
from src.translator import PolicyTranslator

# ── Minimal fake model and tokenizer ─────────────────────────────────────────

class _TinyLM(nn.Module):
    VOCAB = 50

    def __init__(self, seed: int = 42):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embed     = nn.Embedding(self.VOCAB, 16)
            self.down_proj = nn.Linear(16, 16)
            self.up_proj   = nn.Linear(16, 16)
            self.q_proj    = nn.Linear(16, 16)
            self.lm_head   = nn.Linear(16, self.VOCAB)

    def forward(self, input_ids, **_):
        x = torch.relu(self.embed(input_ids))
        x = torch.relu(self.down_proj(x))
        x = self.up_proj(x)
        x = self.q_proj(x)
        return SimpleNamespace(logits=self.lm_head(x))

    def generate(self, input_ids, max_new_tokens=5, pad_token_id=0, **_):
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

    def __init__(self):
        self._id_to_text: dict[tuple, str] = {}

    def __call__(self, text, return_tensors=None, add_special_tokens=True, padding=False, **_):
        ids = [max(2, abs(hash(w)) % 48) for w in text.split()[:6]] or [2]
        self._id_to_text[tuple(ids)] = text
        if return_tensors == "pt":
            t = torch.tensor([ids])
            if padding:
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}
            return {"input_ids": t}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return self._id_to_text.get(tuple(seq), " ".join(str(i) for i in seq if i > 1))


POLICY_TEXT = """\
name: HR Compliance

DENY: salary information
  match: salary, wage, compensation

DENY: personal contact
  match: address, phone, email

ALLOW: general information
"""

GDPR_POLICY_TEXT = """\
name: GDPR Test

DENY: medical information
  severity: critical
  article: 9
  match: medical, health

DENY: home addresses
  severity: high
  article: 4
  match: address, home address

ALLOW: job titles
"""


@pytest.fixture
def model():
    m = _TinyLM()
    wrap_with_nlpn(m, rmax=8, target_modules=["down_proj", "up_proj", "q_proj"])
    m.eval()
    return m


@pytest.fixture
def tokenizer():
    return _FakeTok()


@pytest.fixture
def policy():
    return Policy.from_text(POLICY_TEXT)


# ── 1. Policy parsing ─────────────────────────────────────────────────────────

def test_policy_parses_deny_rules(policy):
    assert len(policy.denied) == 2
    categories = [r.category for r in policy.denied]
    assert "salary information" in categories
    assert "personal contact" in categories


def test_policy_parses_allow_rules(policy):
    assert any(r.action == "ALLOW" for r in policy.rules)


def test_policy_from_file(tmp_path, policy):
    f = tmp_path / "p.txt"
    f.write_text(policy.to_text())
    loaded = Policy.from_file(f)
    assert loaded.name == policy.name
    assert len(loaded.denied) == len(policy.denied)


def test_policy_rule_matches_keyword(policy):
    rule = policy.denied[0]
    assert rule.matches("what is the salary?")
    assert not rule.matches("what are office hours?")


def test_policy_rule_matches_case_insensitive(policy):
    rule = policy.denied[0]
    assert rule.matches("What is the SALARY of the CEO?")


def test_policy_history_multi_turn(policy):
    compiler = PolicyCompiler(policy)
    # Salary mentioned two turns ago — should still trigger
    violated, cats = compiler.check("tell me more", history=["what is the salary", "I see"])
    assert violated
    assert "salary information" in cats


def test_policy_history_beyond_window(policy):
    # Only last 3 turns are checked; salary 4 turns ago should NOT trigger
    compiler = PolicyCompiler(policy)
    history = ["what is the salary", "ok", "next topic", "something else"]
    violated, _ = compiler.check("tell me more", history=history)
    assert not violated


# ── 2. NLPNLinear wrapping ────────────────────────────────────────────────────

def test_wrap_replaces_target_layers(model):
    from src.nlpn import NLPNLinear
    wrapped = {n for n, m in model.named_modules() if isinstance(m, NLPNLinear)}
    assert "down_proj" in wrapped
    assert "q_proj" in wrapped


def test_wrap_preserves_output_at_rmax():
    base   = _TinyLM()
    nlpn   = _TinyLM()  # same seed → same weights
    # 16×16 Linear layers have full rank 16 — rmax must match for exact reconstruction
    wrap_with_nlpn(nlpn, rmax=16, target_modules=["down_proj", "up_proj", "q_proj"])
    nlpn.eval()
    inp = torch.tensor([[5, 10, 15]])
    with torch.no_grad():
        out_base = base(inp).logits
        set_privilege(nlpn, 16)
        out_nlpn = nlpn(inp).logits
    assert torch.allclose(out_base, out_nlpn, atol=1e-4)


def test_privilege_changes_output(model):
    inp = torch.tensor([[5, 10, 15]])
    set_privilege(model, 1)
    with torch.no_grad():
        out_low = model(inp).logits.clone()
    set_privilege(model, 8)
    with torch.no_grad():
        out_high = model(inp).logits
    assert not torch.allclose(out_low, out_high)


def test_nested_structure(model):
    """Im(W(g)) ⊆ Im(W(g+1)) — verified by rank check."""
    from src.nlpn import NLPNLinear
    layer = next(m for m in model.modules() if isinstance(m, NLPNLinear))
    for g in (1, 4, 8):
        W = layer.B[:, :g] @ layer.A[:g, :]
        assert torch.linalg.matrix_rank(W).item() <= g


# ── 3. Policy → privilege allocation ─────────────────────────────────────────

def test_allocator_deny_sets_low_privilege(model, tokenizer, policy):
    compiler  = PolicyCompiler(policy)
    allocator = PolicyAllocator(compiler, tokenizer, low_privilege=1)
    enc = tokenizer("What is the salary?", return_tensors="pt")
    _, g = allocator.generate(model, enc["input_ids"], rmax=8, max_new_tokens=3)
    assert g == 1


def test_allocator_allow_sets_full_privilege(model, tokenizer, policy):
    compiler  = PolicyCompiler(policy)
    allocator = PolicyAllocator(compiler, tokenizer, low_privilege=1)
    enc = tokenizer("What are the office hours?", return_tensors="pt")
    _, g = allocator.generate(model, enc["input_ids"], rmax=8, max_new_tokens=3)
    assert g == 8


# ── 4. Natural language policy translation ───────────────────────────────────

def test_translator_produces_deny_rules():
    policy = PolicyTranslator().translate(
        "Employee salaries and medical records must not be disclosed. "
        "Job titles may be shared."
    )
    assert any(r.action == "DENY" for r in policy.rules)
    assert any(r.action == "ALLOW" for r in policy.rules)


def test_translator_salary_detected():
    policy = PolicyTranslator().translate("Salary information must not be shared.")
    deny_keywords = [kw for r in policy.denied for kw in r.keywords]
    assert any("salary" in kw for kw in deny_keywords)


# ── 5. Training ───────────────────────────────────────────────────────────────

def test_train_runs_and_updates_b(model, tokenizer, policy):
    from src.nlpn import NLPNLinear
    layer = next(m for m in model.modules() if isinstance(m, NLPNLinear))
    b_before = layer.B.data.clone()

    src.train_nlpn(model, tokenizer, policy, config=TrainConfig(epochs=1, log_every=999))

    assert not torch.equal(b_before, layer.B.data), "B matrix should change after training"


def test_train_freezes_a_matrix(model, tokenizer, policy):
    from src.nlpn import NLPNLinear
    layer = next(m for m in model.modules() if isinstance(m, NLPNLinear))
    a_before = layer.A.data.clone()

    src.train_nlpn(model, tokenizer, policy, config=TrainConfig(epochs=1, log_every=999))

    assert torch.equal(a_before, layer.A.data), "A matrix must stay frozen"


def test_train_restores_eval_and_full_privilege(model, tokenizer, policy):
    src.train_nlpn(model, tokenizer, policy, config=TrainConfig(epochs=1, log_every=999))
    assert not model.training
    from src.nlpn import NLPNLinear
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 8


# ── 6. Save / load round-trip ─────────────────────────────────────────────────

def test_save_load_preserves_output(model, tokenizer, tmp_path):
    enc = tokenizer("hello", return_tensors="pt")
    set_privilege(model, 8)
    out_before = model.generate(enc["input_ids"], max_new_tokens=3).clone()

    save_nlpn(model, tmp_path / "ckpt", model_id="test/tiny")

    model2 = _TinyLM()
    wrap_with_nlpn(model2, rmax=8, target_modules=["down_proj", "up_proj", "q_proj"])
    model2.eval()
    load_nlpn(model2, tmp_path / "ckpt")

    set_privilege(model2, 8)
    out_after = model2.generate(enc["input_ids"], max_new_tokens=3)
    assert torch.equal(out_before, out_after)


def test_save_creates_expected_files(model, tmp_path):
    save_nlpn(model, tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "nlpn_weights.pt").exists()
    assert (tmp_path / "ckpt" / "nlpn_config.json").exists()


# ── 7. GDPR tiered enforcement ────────────────────────────────────────────────

def test_gdpr_parser_severity_and_article():
    rules, name = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    medical = next(r for r in rules if r.category == "medical information")
    assert medical.severity == "critical"
    assert medical.article == 9
    assert name == "GDPR Test"


def test_gdpr_allocator_critical_gets_lowest_g(model, tokenizer):
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    allocator = GDPRAllocator(rules, tokenizer)
    enc = tokenizer("What medical conditions does the patient have?", return_tensors="pt")
    g = allocator.allocate(model, enc["input_ids"], rmax=100)
    assert g == 1  # critical = 1% of 100


def test_gdpr_allocator_high_gets_mid_g(model, tokenizer):
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    allocator = GDPRAllocator(rules, tokenizer)
    enc = tokenizer("What is the home address of the CEO?", return_tensors="pt")
    g = allocator.allocate(model, enc["input_ids"], rmax=100)
    assert g == 5  # high = 5% of 100


def test_gdpr_allocator_allow_gets_full_privilege(model, tokenizer):
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    allocator = GDPRAllocator(rules, tokenizer)
    enc = tokenizer("What is the job title of the engineering lead?", return_tensors="pt")
    g = allocator.allocate(model, enc["input_ids"], rmax=100)
    assert g == 100


def test_gdpr_audit_log_records_entry(tmp_path, model, tokenizer):
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    log = AuditLog(tmp_path / "audit.jsonl")
    allocator = GDPRAllocator(rules, tokenizer, audit_log=log)
    enc = tokenizer("What medical conditions does the patient have?", return_tensors="pt")
    allocator.allocate(model, enc["input_ids"], rmax=100)
    assert len(log) == 1
    assert log._entries[0].severity == "critical"


def test_gdpr_audit_log_hmac_verify(tmp_path, model, tokenizer):
    key = b"test-secret-key"
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY_TEXT)
    log = AuditLog(tmp_path / "audit.jsonl", hmac_key=key)
    allocator = GDPRAllocator(rules, tokenizer, audit_log=log)
    enc = tokenizer("What is the home address?", return_tensors="pt")
    allocator.allocate(model, enc["input_ids"], rmax=100)

    result = verify_audit_log(tmp_path / "audit.jsonl", hmac_key=key)
    assert result["valid"] == 1
    assert result["tampered"] == 0


# ── 8. Evaluate and calibrate ────────────────────────────────────────────────

def test_evaluate_nlpn_keys_and_range(model, tokenizer, policy):
    deny_ex = build_deny_examples(policy)
    result  = evaluate_nlpn(model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=8, low_g=1)
    assert set(result) >= {"deny_suppression_rate", "allow_preservation_rate", "low_g", "rmax"}
    assert 0.0 <= result["deny_suppression_rate"]   <= 1.0
    assert 0.0 <= result["allow_preservation_rate"] <= 1.0


def test_calibrate_returns_valid_g(model, tokenizer, policy):
    deny_ex = build_deny_examples(policy)
    low_g   = calibrate_privilege(model, tokenizer, deny_ex, rmax=8)
    assert 1 <= low_g <= 8
    # Model is back at full privilege after calibration
    from src.nlpn import NLPNLinear
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 8
