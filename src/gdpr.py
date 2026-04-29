"""
GDPR-specific privilege enforcement.

Tiered privilege maps GDPR severity to g:
  critical (Art. 9)  → 1% of rmax
  high     (Art. 4)  → 5% of rmax
  medium              → 20% of rmax
  none (ALLOW)        → rmax

Every allocation is recorded to the audit log (Article 30 obligation).
"""
from __future__ import annotations
import hashlib
import hmac as _hmac
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from .policy import Policy, PolicyRule, _BaseAllocator, _parse
from .enforcer import set_privilege, get_rmax

@dataclass
class GDPRRule(PolicyRule):
    """PolicyRule with GDPR-specific severity default."""
    severity: str = "high"  # GDPR default: high (Art. 4 personal data)

SEVERITY_PRIVILEGE: dict[str, float] = {
    "critical": 0.01,
    "high":     0.05,
    "medium":   0.20,
}
_SEVERITY_ORDER = ["critical", "high", "medium"]


@dataclass
class AuditEntry:
    timestamp: str
    prompt_hash: str
    triggered_categories: list[str]
    articles: list[int]
    severity: str
    privilege_granted: int
    rmax: int

    @property
    def privilege_pct(self) -> float:
        return round(100 * self.privilege_granted / self.rmax, 1)


class AuditLog:
    """Article 30 audit log. Stores prompt hashes, not raw text (data minimisation)."""

    def __init__(
        self,
        path: str | Path = "audit/gdpr_audit.jsonl",
        max_entries: int | None = None,
        hmac_key: bytes | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._hmac_key = hmac_key
        self._entries: list[AuditEntry] = []

    def _sign(self, row: dict) -> str:
        msg = json.dumps(row, sort_keys=True).encode()
        return _hmac.new(self._hmac_key, msg, "sha256").hexdigest()

    def _rotate(self) -> None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        self.path.rename(self.path.with_name(f"{self.path.stem}_{ts}{self.path.suffix}"))
        self._entries.clear()

    def record(self, entry: AuditEntry) -> None:
        if self.max_entries and len(self._entries) >= self.max_entries:
            self._rotate()
        self._entries.append(entry)
        row = asdict(entry)
        if self._hmac_key:
            row["_hmac"] = self._sign(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def summary(self) -> str:
        if not self._entries:
            return "Audit log: empty"
        suppressed = [e for e in self._entries if e.privilege_granted < e.rmax]
        lines = [f"Audit log: {len(self._entries)} entries  "
                 f"({len(suppressed)} suppressed, {len(self._entries)-len(suppressed)} full)"]
        cat_counts: dict[str, int] = {}
        for e in suppressed:
            for cat in e.triggered_categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:<40} {n} suppressed requests")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)


class GDPRPolicyParser:
    """Parse a GDPR policy file (superset of base format — adds severity/article)."""

    @staticmethod
    def parse(text: str) -> tuple[list[PolicyRule], str]:
        policy = _parse(text)
        name = policy.name if policy.name != "Unnamed Policy" else "GDPR Policy"
        # Patch DENY rules: default severity to "high" (GDPR Art. 4) if unset
        for rule in policy.rules:
            if rule.action == "DENY" and rule.severity == "none":
                rule.severity = "high"
        return policy.rules, name


class GDPRAllocator(_BaseAllocator):
    """
    Allocator that maps GDPR severity tiers to privilege levels and writes an audit log.
    """

    def __init__(
        self,
        rules: list[PolicyRule],
        tokenizer,
        audit_log: AuditLog | None = None,
        severity_privilege: dict[str, float] | None = None,
    ):
        self.rules = [r for r in rules if r.action == "DENY"]
        self.tokenizer = tokenizer
        self.audit_log = audit_log or AuditLog()
        self.severity_privilege = severity_privilege or SEVERITY_PRIVILEGE

    def allocate(self, model, input_ids: torch.Tensor, rmax: int, **_) -> int:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        triggered = [r for r in self.rules if r.matches(text)]

        if not triggered:
            self._log(text, [], [], "none", rmax, rmax)
            return rmax

        worst = min(
            triggered,
            key=lambda r: _SEVERITY_ORDER.index(r.severity)
            if r.severity in _SEVERITY_ORDER else len(_SEVERITY_ORDER),
        )
        frac = self.severity_privilege.get(worst.severity, 0.05)
        g = max(1, int(rmax * frac))

        categories = [r.category for r in triggered]
        articles   = sorted({r.article for r in triggered if r.article})
        self._log(text, categories, articles, worst.severity, g, rmax)
        arts_str = f"Art.{articles}" if articles else ""
        print(f"  [GDPR DENY] {worst.severity.upper()} {arts_str}  "
              f"→ {categories}  →  g={g}/{rmax} ({frac*100:.0f}%)")
        return g

    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rmax: int | None = None,
        **generate_kwargs,
    ) -> tuple[torch.Tensor, int]:
        if rmax is None:
            rmax = get_rmax(model)
        g = self.allocate(model, input_ids, rmax)
        set_privilege(model, g)
        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id, **generate_kwargs,
            )
        return output, g

    def _log(self, text, categories, articles, severity, g, rmax) -> None:
        self.audit_log.record(AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            prompt_hash=hashlib.sha256(text.encode()).hexdigest(),
            triggered_categories=categories,
            articles=articles,
            severity=severity,
            privilege_granted=g,
            rmax=rmax,
        ))


def verify_audit_log(path: str | Path, hmac_key: bytes) -> dict:
    """Verify integrity of an HMAC-signed audit log."""
    path = Path(path)
    total = valid = tampered = unsigned = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                tampered += 1
                continue
            stored = row.pop("_hmac", None)
            if stored is None:
                unsigned += 1
                continue
            expected = _hmac.new(
                hmac_key, json.dumps(row, sort_keys=True).encode(), "sha256"
            ).hexdigest()
            if _hmac.compare_digest(stored, expected):
                valid += 1
            else:
                tampered += 1
    return {"total": total, "valid": valid, "tampered": tampered, "unsigned": unsigned}
