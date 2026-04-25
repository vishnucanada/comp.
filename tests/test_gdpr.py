import re
import pytest
from src.policy import PolicyRule
from src.gdpr import (
    GDPRRule,
    GDPRPolicyParser,
    AuditEntry,
    AuditLog,
    SEVERITY_PRIVILEGE,
)


GDPR_POLICY = """\
name: Test GDPR Policy

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


# ── GDPRRule ──────────────────────────────────────────────────────────────────

def test_gdpr_rule_is_policy_rule():
    assert issubclass(GDPRRule, PolicyRule)

def test_gdpr_rule_inherits_matches():
    rule = GDPRRule(action="DENY", category="names", keywords=["john"], article=4, severity="high")
    assert rule.matches("What is john's address?")
    assert not rule.matches("What is 2+2?")

def test_gdpr_rule_pattern_matching():
    rule = GDPRRule(
        action="DENY",
        category="ssn",
        patterns=[re.compile(r"\b\d{3}-\d{2}-\d{4}\b")],
        article=9,
        severity="critical",
    )
    assert rule.matches("SSN: 123-45-6789")
    assert not rule.matches("Phone: 555-1234")

def test_gdpr_rule_defaults():
    rule = GDPRRule(action="DENY", category="test")
    assert rule.article is None
    assert rule.severity == "high"


# ── GDPRPolicyParser ──────────────────────────────────────────────────────────

def test_parser_name():
    rules, name = GDPRPolicyParser.parse(GDPR_POLICY)
    assert name == "Test GDPR Policy"

def test_parser_deny_count():
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY)
    deny = [r for r in rules if r.action == "DENY"]
    assert len(deny) == 2

def test_parser_severity():
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY)
    medical = next(r for r in rules if r.category == "medical information")
    assert medical.severity == "critical"

def test_parser_article():
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY)
    medical = next(r for r in rules if r.category == "medical information")
    assert medical.article == 9

def test_parser_keywords():
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY)
    medical = next(r for r in rules if r.category == "medical information")
    assert "medical" in medical.keywords
    assert "health" in medical.keywords

def test_parser_allow():
    rules, _ = GDPRPolicyParser.parse(GDPR_POLICY)
    allow = [r for r in rules if r.action == "ALLOW"]
    assert len(allow) == 1
    assert allow[0].category == "job titles"

def test_parser_invalid_article_ignored():
    text = "DENY: test\n  article: notanumber\n  match: kw\n"
    rules, _ = GDPRPolicyParser.parse(text)
    assert rules[0].article is None

def test_parser_invalid_regex_ignored():
    text = "DENY: test\n  regex: [invalid\n  match: kw\n"
    rules, _ = GDPRPolicyParser.parse(text)
    assert rules[0].patterns == []

def test_parser_default_name():
    rules, name = GDPRPolicyParser.parse("DENY: cat\n  match: kw\n")
    assert name == "GDPR Policy"


# ── AuditEntry ────────────────────────────────────────────────────────────────

def test_audit_entry_privilege_pct():
    entry = AuditEntry(
        timestamp="2024-01-01T00:00:00",
        prompt_hash="abc",
        triggered_categories=["names"],
        articles=[4],
        severity="high",
        privilege_granted=5,
        rmax=100,
    )
    assert entry.privilege_pct == 5.0

def test_audit_entry_full_privilege_pct():
    entry = AuditEntry(
        timestamp="2024-01-01T00:00:00",
        prompt_hash="abc",
        triggered_categories=[],
        articles=[],
        severity="none",
        privilege_granted=100,
        rmax=100,
    )
    assert entry.privilege_pct == 100.0


# ── AuditLog ──────────────────────────────────────────────────────────────────

def _make_entry(severity="high", g=5, rmax=100, cats=("names",)):
    return AuditEntry(
        timestamp="2024-01-01T00:00:00",
        prompt_hash="deadbeef",
        triggered_categories=list(cats),
        articles=[4],
        severity=severity,
        privilege_granted=g,
        rmax=rmax,
    )

def test_audit_log_record(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_make_entry())
    assert len(log) == 1

def test_audit_log_written_to_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record(_make_entry())
    content = path.read_text()
    assert "names" in content
    assert "deadbeef" in content

def test_audit_log_summary_empty(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    assert log.summary() == "Audit log: empty"

def test_audit_log_summary_counts(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_make_entry(cats=("names",)))
    log.record(_make_entry(cats=("names", "emails")))
    log.record(_make_entry(g=100, rmax=100, cats=()))  # full privilege — not suppressed
    summary = log.summary()
    assert "3 entries" in summary
    assert "2 suppressed" in summary
    assert "names" in summary

def test_audit_log_creates_parent_dir(tmp_path):
    log = AuditLog(tmp_path / "subdir" / "audit.jsonl")
    log.record(_make_entry())
    assert (tmp_path / "subdir" / "audit.jsonl").exists()
