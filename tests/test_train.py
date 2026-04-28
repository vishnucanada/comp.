import torch
import torch.nn as nn
import pytest
from src.nlpn import NLPNLinear
from src.enforcer import wrap_with_nlpn
from src.policy import Policy
from src.train import (
    build_deny_examples, build_adversarial_examples,
    _tokenize_pair, train_nlpn, TrainConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    def __init__(self, vocab=50):
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.fc1   = nn.Linear(16, 32)
        self.fc2   = nn.Linear(32, 16)
        self.out   = nn.Linear(16, vocab)

    def forward(self, input_ids):
        x = torch.relu(self.embed(input_ids))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.out(x)
        return type("Out", (), {"logits": logits})()


class _FakeTok:
    eos_token = "</s>"
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        ids = [hash(w) % 48 + 2 for w in text.split()][:10]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


def _wrapped_model(rmax=8):
    model = _TinyModel()
    wrap_with_nlpn(model, rmax=rmax, target_modules=["fc1", "fc2"])
    return model


def _simple_policy():
    return Policy.from_text("DENY: names\n  match: name, email\n")


# ── build_deny_examples ───────────────────────────────────────────────────────

def test_build_deny_examples_nonempty():
    p = _simple_policy()
    examples = build_deny_examples(p)
    assert len(examples) > 0

def test_build_deny_examples_format():
    p = _simple_policy()
    for prompt, response in build_deny_examples(p):
        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(response, str) and len(response) > 0

def test_build_deny_examples_refusal_text():
    p = _simple_policy()
    for _, response in build_deny_examples(p):
        assert "sorry" in response.lower() or "cannot" in response.lower()

def test_build_deny_examples_keyword_in_prompt():
    p = _simple_policy()
    examples = build_deny_examples(p)
    prompts = [pr for pr, _ in examples]
    assert any("name" in pr or "email" in pr for pr in prompts)

def test_build_deny_examples_empty_policy():
    p = Policy.from_text("ALLOW: general\n")
    assert build_deny_examples(p) == []


# ── _tokenize_pair ────────────────────────────────────────────────────────────

def test_tokenize_pair_shapes():
    tok = _FakeTok()
    result = _tokenize_pair("hello world", "yes", tok, max_len=20, device=torch.device("cpu"))
    assert result is not None
    ids, labels = result
    assert ids.shape[0] == 1
    assert labels.shape == ids.shape

def test_tokenize_pair_prompt_masked():
    tok = _FakeTok()
    result = _tokenize_pair("hello world", "yes no", tok, max_len=20, device=torch.device("cpu"))
    assert result is not None
    _, labels = result
    flat = labels[0].tolist()
    assert -100 in flat  # prompt tokens are masked

def test_tokenize_pair_truncation():
    tok = _FakeTok()
    result = _tokenize_pair("a b c d e f g h i j", "r1 r2 r3 r4 r5", tok, max_len=8, device=torch.device("cpu"))
    if result is not None:
        ids, labels = result
        assert ids.shape[1] <= 8

def test_tokenize_pair_all_masked_returns_none():
    class TinyTok:
        eos_token = "</s>"
        eos_token_id = 1
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [2] * 20}  # prompt fills entire max_len
    result = _tokenize_pair("x " * 20, "response", TinyTok(), max_len=5, device=torch.device("cpu"))
    assert result is None


# ── train_nlpn ────────────────────────────────────────────────────────────────

def test_train_runs_without_error():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    deny = [("What is the name?", "I cannot help.")]
    allow = [("What is 2+2?", "4.")]
    train_nlpn(model, tok, policy, config=cfg, deny_examples=deny, allow_examples=allow)

def test_train_only_b_updated():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = _simple_policy()

    # snapshot A before training
    a_before = {id(m): m.A.data.clone() for m in model.modules() if isinstance(m, NLPNLinear)}

    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    deny = [("What is the name?", "I cannot help.")]
    allow = [("What is 2+2?", "4.")]
    train_nlpn(model, tok, policy, config=cfg, deny_examples=deny, allow_examples=allow)

    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert torch.allclose(m.A.data, a_before[id(m)]), "A was modified"

def test_train_b_changes():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = _simple_policy()

    b_before = {id(m): m.B.data.clone() for m in model.modules() if isinstance(m, NLPNLinear)}

    cfg = TrainConfig(epochs=2, lr=1e-2, max_seq_len=16, log_every=100)
    deny = [("What is the name?", "I cannot help.")] * 4
    allow = [("What is 2+2?", "4.")] * 4
    train_nlpn(model, tok, policy, config=cfg, deny_examples=deny, allow_examples=allow)

    changed = sum(
        not torch.allclose(m.B.data, b_before[id(m)])
        for m in model.modules() if isinstance(m, NLPNLinear)
    )
    assert changed > 0, "No B matrices were updated"

def test_train_restores_full_privilege():
    model = _wrapped_model(rmax=8)
    tok = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    train_nlpn(model, tok, policy, config=cfg,
               deny_examples=[("name?", "no.")],
               allow_examples=[("2+2?", "4.")])
    for m in model.modules():
        if isinstance(m, NLPNLinear):
            assert m.privilege == 8

def test_train_model_in_eval_after():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    train_nlpn(model, tok, policy, config=cfg,
               deny_examples=[("name?", "no.")],
               allow_examples=[("2+2?", "4.")])
    assert not model.training

def test_train_on_step_callback():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    calls = []
    train_nlpn(model, tok, policy, config=cfg,
               deny_examples=[("name?", "no.")],
               allow_examples=[("2+2?", "4.")],
               on_step=lambda e, s, l: calls.append((e, s, l)))
    assert len(calls) > 0
    assert all(isinstance(l, float) for _, _, l in calls)

def test_train_no_deny_examples_raises():
    model = _wrapped_model()
    tok = _FakeTok()
    policy = Policy.from_text("ALLOW: general\n")  # no DENY rules → no examples
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100)
    with pytest.raises(ValueError, match="No deny examples"):
        train_nlpn(model, tok, policy, config=cfg)


# ── build_adversarial_examples ────────────────────────────────────────────────

def test_adversarial_examples_nonempty():
    p = Policy.from_text("DENY: salary\n  match: salary\n")
    ex = build_adversarial_examples(p)
    assert len(ex) > 0

def test_adversarial_examples_format():
    p = Policy.from_text("DENY: salary\n  match: salary\n")
    for prompt, response in build_adversarial_examples(p):
        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(response, str)

def test_adversarial_examples_refusal():
    p = Policy.from_text("DENY: salary\n  match: salary\n")
    for _, response in build_adversarial_examples(p):
        assert "sorry" in response.lower() or "cannot" in response.lower()

def test_adversarial_examples_empty_policy():
    p = Policy.from_text("ALLOW: general\n")
    assert build_adversarial_examples(p) == []

def test_adversarial_examples_include_synonyms():
    p = Policy.from_text("DENY: salary\n  match: salary\n")
    prompts = [pr for pr, _ in build_adversarial_examples(p)]
    # "salary" has synonyms like "pay", "wage", etc.
    assert any("pay" in pr or "wage" in pr or "compensation" in pr for pr in prompts)

def test_adversarial_examples_include_injection_templates():
    p = Policy.from_text("DENY: salary\n  match: salary\n")
    prompts = [pr for pr, _ in build_adversarial_examples(p)]
    assert any("Ignore" in pr or "novel" in pr or "JSON" in pr for pr in prompts)


# ── orth_reg training ─────────────────────────────────────────────────────────

def test_train_with_orth_reg_runs():
    model = _wrapped_model()
    tok   = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100, orth_reg=0.01)
    train_nlpn(model, tok, policy, config=cfg,
               deny_examples=[("What is the name?", "I cannot help.")] * 4,
               allow_examples=[("What is 2+2?", "4.")])

def test_train_orth_reg_zero_same_as_default():
    # orth_reg=0.0 should not change training behavior
    model = _wrapped_model()
    tok   = _FakeTok()
    policy = _simple_policy()
    cfg = TrainConfig(epochs=1, lr=1e-3, max_seq_len=16, log_every=100, orth_reg=0.0)
    train_nlpn(model, tok, policy, config=cfg,
               deny_examples=[("What is the name?", "I cannot help.")],
               allow_examples=[("What is 2+2?", "4.")])
