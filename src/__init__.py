"""comp. — policy-as-code for LLM API calls.

A small enforcement layer for any LLM backend. Define DENY/ALLOW rules in a
text policy, plug in a content guard (keyword, OpenAI moderation, Llama Guard),
optionally attach IAM roles, and gate every call through PolicyGate. Every
decision goes to an append-only JSONL audit log.

Quick start
-----------
    from src import IAMConfig, KeywordGuard, Policy, PolicyGate

    policy = Policy.from_file("policies/hr.txt")
    iam = IAMConfig.from_yaml("iam.yaml")
    gate = PolicyGate(policy, guard=KeywordGuard(policy), iam=iam,
                      audit_log_path="audit/gate.jsonl")

    response, decision = gate.complete(
        message="What is John's salary?",
        fn=lambda m: claude.messages.create(...).content[0].text,
        user_role="anonymous",
    )
"""

from .gate import GateDecision, PolicyGate
from .guard import (
    CompositeGuard,
    Guard,
    GuardResult,
    KeywordGuard,
    LlamaGuardGuard,
    OpenAIModerationGuard,
    make_guard,
)
from .iam import IAMConfig, Role
from .policy import Policy, PolicyCompiler
from .report import ComplianceReport, generate_report

__all__ = [
    "ComplianceReport",
    "CompositeGuard",
    "GateDecision",
    "Guard",
    "GuardResult",
    "IAMConfig",
    "KeywordGuard",
    "LlamaGuardGuard",
    "OpenAIModerationGuard",
    "Policy",
    "PolicyCompiler",
    "PolicyGate",
    "Role",
    "generate_report",
    "make_guard",
]
