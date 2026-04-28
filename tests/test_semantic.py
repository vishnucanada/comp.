"""
Tests for SemanticPolicyCompiler.

sentence-transformers is optional; all tests run regardless of whether it is
installed.  Tests that verify semantic-specific behaviour are skipped when the
library is absent.
"""
import pytest
from src.policy import Policy
from src.semantic import SemanticPolicyCompiler

try:
    import sentence_transformers  # noqa: F401
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

_SKIP_NO_ST = pytest.mark.skipif(not _ST_AVAILABLE, reason="sentence-transformers not installed")


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _policy():
    return Policy.from_text(
        "DENY: salary information\n  match: salary, wage\n"
        "DENY: medical information\n  match: medical, health\n"
    )


# ── Always-present behaviour (keyword fallback) ───────────────────────────────

def test_keyword_match_always_works():
    c = SemanticPolicyCompiler(_policy())
    violated, cats = c.check("What is the employee salary?")
    assert violated
    assert "salary information" in cats

def test_allow_always_works():
    c = SemanticPolicyCompiler(_policy())
    violated, _ = c.check("What are office hours?")
    assert not violated

def test_semantic_active_property():
    c = SemanticPolicyCompiler(_policy())
    assert isinstance(c.semantic_active, bool)

def test_history_with_keyword():
    c = SemanticPolicyCompiler(_policy())
    violated, cats = c.check("tell me more", history=["what is the salary?"])
    assert violated
    assert "salary information" in cats


# ── Semantic-only behaviour (requires sentence-transformers) ──────────────────

@_SKIP_NO_ST
def test_semantic_active_when_st_installed():
    c = SemanticPolicyCompiler(_policy())
    assert c.semantic_active is True

@_SKIP_NO_ST
def test_semantic_catches_paraphrase():
    c = SemanticPolicyCompiler(_policy(), threshold=0.4)
    # "pay rate" doesn't contain "salary" or "wage" literally
    violated, cats = c.check("What is the employee pay rate?")
    assert violated
    assert "salary information" in cats

@_SKIP_NO_ST
def test_high_threshold_does_not_over_trigger():
    c = SemanticPolicyCompiler(_policy(), threshold=0.99)
    # Very high threshold — only exact keyword matches should fire
    violated, _ = c.check("What is the weather today?")
    assert not violated

@_SKIP_NO_ST
def test_semantic_deduplicates_with_keyword_match():
    c = SemanticPolicyCompiler(_policy())
    _, cats = c.check("What is the salary and wage?")
    # Both keyword and semantic paths catch "salary information" — should appear once
    assert cats.count("salary information") == 1
