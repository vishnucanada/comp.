import pytest
from src.translator import (
    PolicyTranslator,
    _split_sentences,
    _matched_categories,
    _infer_name,
    _text_has,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_split_sentences_period():
    assert _split_sentences("Hello world. How are you?") == ["Hello world.", "How are you?"]

def test_split_sentences_newline():
    parts = _split_sentences("Line one.\nLine two.")
    assert len(parts) == 2

def test_split_sentences_empty():
    assert _split_sentences("") == []

def test_text_has_multiword():
    assert _text_has("email address here", "email address")
    assert not _text_has("just email here", "email address")

def test_text_has_word_boundary():
    assert _text_has("my ssn is 123", "ssn")
    assert not _text_has("my passn is 123", "ssn")

def test_matched_categories_email():
    cats = _matched_categories("Do not share email addresses.")
    names = [c["name"] for c in cats]
    assert "email addresses" in names

def test_matched_categories_salary():
    cats = _matched_categories("salary information must be kept confidential")
    names = [c["name"] for c in cats]
    assert "salary information" in names

def test_matched_categories_no_match():
    cats = _matched_categories("The sky is blue today.")
    assert cats == []

def test_infer_name_from_subject():
    text = "Subject: HR Data Policy\nDo not share names."
    assert _infer_name(text) == "HR Data Policy"

def test_infer_name_from_policy_prefix():
    text = "Policy: Employee Privacy\nNames must not be shared."
    assert _infer_name(text) == "Employee Privacy"

def test_infer_name_first_short_line():
    text = "Short title\nA much longer line that goes on and on and on forever."
    assert _infer_name(text) == "Short title"

def test_infer_name_fallback():
    text = "a" * 100
    assert _infer_name(text) == "Extracted Policy"


# ── PolicyTranslator ──────────────────────────────────────────────────────────

def test_translate_deny_names():
    t = PolicyTranslator()
    p = t.translate("Employee names must not be shared with third parties.")
    categories = [r.category for r in p.denied]
    assert "names" in categories

def test_translate_deny_email():
    t = PolicyTranslator()
    p = t.translate("Email addresses are strictly confidential.")
    categories = [r.category for r in p.denied]
    assert "email addresses" in categories

def test_translate_deny_medical():
    t = PolicyTranslator()
    p = t.translate("Medical records must not be disclosed.")
    categories = [r.category for r in p.denied]
    assert "medical information" in categories

def test_translate_always_has_allow_rule():
    t = PolicyTranslator()
    p = t.translate("Passwords must not be shared.")
    assert any(r.action == "ALLOW" for r in p.rules)

def test_translate_allow_explicit():
    t = PolicyTranslator()
    p = t.translate(
        "Do not share names. "
        "Job titles are publicly available."
    )
    allow_cats = [r.category for r in p.allowed]
    assert len(allow_cats) >= 1

def test_translate_multiple_deny():
    t = PolicyTranslator()
    p = t.translate(
        "Email addresses must not be shared. "
        "Medical records must not be disclosed."
    )
    deny_cats = [r.category for r in p.denied]
    assert "email addresses" in deny_cats
    assert "medical information" in deny_cats

def test_translate_deny_has_keywords():
    t = PolicyTranslator()
    p = t.translate("Do not share email addresses.")
    rule = next(r for r in p.denied if r.category == "email addresses")
    assert len(rule.keywords) > 0

def test_translate_deny_has_patterns_for_email():
    t = PolicyTranslator()
    p = t.translate("Do not share email addresses.")
    rule = next(r for r in p.denied if r.category == "email addresses")
    assert len(rule.patterns) > 0
    assert rule.patterns[0].search("user@example.com")

def test_translate_and_save(tmp_path):
    t = PolicyTranslator()
    out = tmp_path / "out.txt"
    p = t.translate_and_save("Email addresses must not be shared.", out)
    assert out.exists()
    content = out.read_text()
    assert "DENY" in content
    assert "email addresses" in content
