"""Compliance report generator.

Combines policy summary with audit-log statistics into an auditor-readable
document (Markdown or JSON). Reads JSONL audit logs produced by PolicyGate.

Scope: this report is a useful operational artefact for review, not a
compliance certification. It documents what the gate did; it does not attest
that the gate's policy is itself sufficient under any specific regulation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ComplianceReport:
    policy_name: str
    generated_at: str
    policy_rules: list[dict[str, Any]] = field(default_factory=list)
    audit_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "generated_at": self.generated_at,
            "policy_rules": self.policy_rules,
            "audit_stats": self.audit_stats,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Compliance Report: {self.policy_name}",
            f"_Generated: {self.generated_at}_",
            "",
            "## Policy",
        ]

        deny_rules = [r for r in self.policy_rules if r.get("action") == "DENY"]
        allow_rules = [r for r in self.policy_rules if r.get("action") == "ALLOW"]

        if deny_rules:
            lines.append("")
            lines.append("**Denied categories:**")
            for r in deny_rules:
                kws = ", ".join(r.get("keywords", [])[:5])
                lines.append(f"- **{r['category']}** — `{kws}`")

        if allow_rules:
            lines.append("")
            lines.append("**Allowed categories:**")
            for r in allow_rules:
                lines.append(f"- {r['category']}")

        if self.audit_stats:
            lines += ["", "## Audit Log", ""]
            for k, v in self.audit_stats.items():
                if isinstance(v, dict):
                    lines.append(f"- **{k}**:")
                    for sub_k, sub_v in v.items():
                        lines.append(f"  - {sub_k}: {sub_v}")
                else:
                    lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)


def generate_report(
    policy,
    audit_log_path: str | Path | None = None,
) -> ComplianceReport:
    """Build a ComplianceReport for the given policy.

    Args:
        policy: Policy object, or a str/Path to a policy file.
        audit_log_path: JSONL audit log path. If provided, includes usage statistics.
    """
    from .policy import Policy

    if isinstance(policy, (str, Path)):
        policy = Policy.from_file(policy)

    rules = [
        {"action": r.action, "category": r.category, "keywords": r.keywords}
        for r in policy.rules
    ]

    audit_stats: dict[str, Any] = {}
    if audit_log_path:
        audit_stats = _summarize_audit_log(audit_log_path)

    return ComplianceReport(
        policy_name=policy.name,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        policy_rules=rules,
        audit_stats=audit_stats,
    )


def _summarize_audit_log(path: str | Path) -> dict[str, Any]:
    """Read a JSONL gate audit log and return summary statistics."""
    path = Path(path)
    if not path.exists():
        return {"error": f"{path} not found"}

    total = denied = 0
    cat_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if not row.get("allowed", True):
                denied += 1
            for cat in row.get("categories", []):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            role = row.get("user_role")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
            backend = row.get("backend")
            if backend:
                backend_counts[backend] = backend_counts.get(backend, 0) + 1

    stats: dict[str, Any] = {
        "total_requests": total,
        "denied_requests": denied,
        "allow_rate": f"{(total - denied) / total:.0%}" if total else "N/A",
    }
    if cat_counts:
        top = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
        stats["top_triggered_categories"] = dict(top)
    if role_counts:
        stats["requests_by_role"] = role_counts
    if backend_counts:
        stats["decisions_by_backend"] = backend_counts
    return stats
