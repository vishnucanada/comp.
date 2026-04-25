import re
import pytest
from src.policy import PolicyRule, Policy, PolicyCompiler, _parse


# ── PolicyRule ────────────────────────────────────────────────────────────────

def test_rule_matches_keyword():
    rule = PolicyRule("DENY", "names", keywords=["john", "full name"])
    assert rule.matches("What is John's address?")
    assert rule.matches("Give me the full name")
    assert not rule.matches("What is the weather?")

def test_rule_matches_case_insensitive():
    rule = PolicyRule("DENY", "names", keywords=["email"])
    assert rule.matches("WHAT IS THE EMAIL?")
    assert rule.matches("email address")

def test_rule_matches_pattern():
    rule = PolicyRule("DENY", "ssn", patterns=[re.compile(r"\b\d{3}-\d{2}-\d{4}\b")])
    assert rule.matches("My SSN is 123-45-6789")
    assert not rule.matches("Call me at 555-1234")

def test_rule_matches_no_keywords_or_patterns():
    rule = PolicyRule("ALLOW", "general")
    assert not rule.matches("anything")

def test_rule_repr():
    rule = PolicyRule("DENY", "names")
    assert "DENY" in repr(rule)
    assert "names" in repr(rule)


# ── Policy parsing ────────────────────────────────────────────────────────────

POLICY_TEXT = """\
# test policy
name: Test Policy

DENY: names
  match: name, full name
  regex: \\bjohn\\b

ALLOW: general information
"""

def test_parse_name():
    p = Policy.from_text(POLICY_TEXT)
    assert p.name == "Test Policy"

def test_parse_rules():
    p = Policy.from_text(POLICY_TEXT)
    assert len(p.rules) == 2

def test_parse_deny_rule():
    p = Policy.from_text(POLICY_TEXT)
    deny = p.denied
    assert len(deny) == 1
    assert deny[0].category == "names"
    assert "name" in deny[0].keywords
    assert "full name" in deny[0].keywords

def test_parse_deny_regex():
    p = Policy.from_text(POLICY_TEXT)
    assert len(p.denied[0].patterns) == 1
    assert p.denied[0].patterns[0].search("call john now")

def test_parse_allow_rule():
    p = Policy.from_text(POLICY_TEXT)
    allow = p.allowed
    assert len(allow) == 1
    assert allow[0].category == "general information"

def test_parse_comments_ignored():
    p = Policy.from_text("# comment\nname: X\nDENY: cat\n  match: kw\n")
    assert p.name == "X"
    assert len(p.rules) == 1

def test_parse_invalid_regex_is_skipped(capsys):
    p = Policy.from_text("DENY: cat\n  regex: [invalid\n  match: kw\n")
    assert len(p.denied[0].patterns) == 0
    assert "Warning" in capsys.readouterr().out

def test_parse_default_name():
    p = Policy.from_text("DENY: cat\n  match: kw\n")
    assert p.name == "Unnamed Policy"

def test_from_file(tmp_path):
    f = tmp_path / "pol.txt"
    f.write_text("name: File Policy\nDENY: x\n  match: kw\n")
    p = Policy.from_file(f)
    assert p.name == "File Policy"


# ── Policy.to_text round-trip ─────────────────────────────────────────────────

def test_to_text_contains_name():
    p = Policy.from_text(POLICY_TEXT)
    assert "name: Test Policy" in p.to_text()

def test_to_text_contains_rules():
    p = Policy.from_text(POLICY_TEXT)
    txt = p.to_text()
    assert "DENY: names" in txt
    assert "ALLOW: general information" in txt
    assert "match: name" in txt

def test_to_text_round_trips():
    p1 = Policy.from_text(POLICY_TEXT)
    p2 = Policy.from_text(p1.to_text())
    assert p2.name == p1.name
    assert len(p2.rules) == len(p1.rules)
    assert p2.denied[0].keywords == p1.denied[0].keywords


# ── PolicyCompiler ────────────────────────────────────────────────────────────

def _make_compiler() -> PolicyCompiler:
    p = Policy.from_text(POLICY_TEXT)
    return PolicyCompiler(p)

def test_compiler_deny():
    c = _make_compiler()
    violated, cats = c.check("What is John's full name?")
    assert violated
    assert "names" in cats

def test_compiler_allow():
    c = _make_compiler()
    violated, cats = c.check("What is the weather today?")
    assert not violated
    assert cats == []

def test_compiler_multiple_rules():
    p = Policy.from_text(
        "DENY: names\n  match: name\n"
        "DENY: emails\n  match: email\n"
    )
    c = PolicyCompiler(p)
    violated, cats = c.check("name and email please")
    assert violated
    assert "names" in cats
    assert "emails" in cats

def test_compiler_repr():
    c = _make_compiler()
    assert "names" in repr(c)
