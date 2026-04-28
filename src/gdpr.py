"""
GDPR-specific privilege enforcement.

Two additions over the base PolicyAllocator:

1. Tiered privilege — severity level in the policy maps to different g values:
     critical (Art. 9)  → g = 1%  of rmax   (special categories)
     high     (Art. 4)  → g = 5%  of rmax   (standard personal data)
     medium              → g = 20% of rmax   (indirect / pseudonymised)
     none (ALLOW)        → g = rmax          (full privilege)

2. Audit log — every allocation decision is recorded (Article 30 obligation):
     timestamp, prompt hash, categories triggered, article, privilege granted
"""
from __future__ import annotations
import hashlib
import hmac as _hmac
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from .policy import PolicyRule
from .enforcer import set_privilege, get_rmax


# Fraction of rmax assigned to each severity tier
SEVERITY_PRIVILEGE: dict[str, float] = {
    "critical": 0.01,
    "high":     0.05,
    "medium":   0.20,
}

_SEVERITY_ORDER = ["critical", "high", "medium"]


@dataclass
class GDPRRule(PolicyRule):
    """PolicyRule extended with GDPR article number and severity."""
    article: int | None = None
    severity: str = "high"  # critical | high | medium | none


@dataclass
class AuditEntry:
    timestamp: str
    prompt_hash: str          # SHA-256 of prompt — no raw text stored (data minimisation)
    triggered_categories: list[str]
    articles: list[int]
    severity: str
    privilege_granted: int
    rmax: int

    @property
    def privilege_pct(self) -> float:
        return round(100 * self.privilege_granted / self.rmax, 1)


class AuditLog:
    """
    Article 30 audit log — records of processing activities.
    Stores prompt hashes, not raw prompts, to comply with data minimisation.

    Args:
        path:        Path to the JSONL audit file.
        max_entries: Rotate (archive) the log file when this many entries
                     are reached. None = never rotate.
        hmac_key:    Optional bytes key. When provided, each written row
                     includes an ``_hmac`` field (HMAC-SHA256 of the row)
                     so tamper-evident verification is possible.
    """

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
        """Archive the current log file and start a new one."""
        ts = time.strftime("%Y%m%dT%H%M%S")
        archive = self.path.with_name(f"{self.path.stem}_{ts}{self.path.suffix}")
        self.path.rename(archive)
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
        lines = [
            f"Audit log: {len(self._entries)} entries  "
            f"({len(suppressed)} suppressed, {len(self._entries)-len(suppressed)} full)",
        ]
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
    """Parse a GDPR policy file (superset of base policy format, adds severity/article fields)."""

    @staticmethod
    def parse(text: str) -> tuple[list[GDPRRule], str]:
        name = "GDPR Policy"
        rules: list[GDPRRule] = []
        current: GDPRRule | None = None

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("name:"):
                name = line.split(":", 1)[1].strip()

            elif line.upper().startswith("DENY:"):
                if current:
                    rules.append(current)
                current = GDPRRule(
                    action="DENY",
                    category=line.split(":", 1)[1].strip(),
                    article=None,
                    severity="high",
                )

            elif line.upper().startswith("ALLOW:"):
                if current:
                    rules.append(current)
                current = GDPRRule(
                    action="ALLOW",
                    category=line.split(":", 1)[1].strip(),
                    article=None,
                    severity="none",
                )

            elif current and line.lower().startswith("severity:"):
                current.severity = line.split(":", 1)[1].strip().lower()

            elif current and line.lower().startswith("article:"):
                try:
                    current.article = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif current and line.lower().startswith("match:"):
                kws = [k.strip().lower() for k in line.split(":", 1)[1].split(",")]
                current.keywords.extend(k for k in kws if k)

            elif current and line.lower().startswith("regex:"):
                pattern_str = line.split(":", 1)[1].strip()
                try:
                    current.patterns.append(re.compile(pattern_str))
                except re.error:
                    pass

        if current:
            rules.append(current)
        return rules, name


class GDPRAllocator:
    """
    Allocator that maps GDPR severity tiers to privilege levels.

    Severity → privilege fraction of rmax:
      critical  (Art. 9 special categories)  →  1%
      high      (Art. 4 personal data)        →  5%
      medium    (pseudonymised/indirect)       → 20%
      no match                                 → 100%

    Also records every allocation decision to the audit log.
    """

    def __init__(
        self,
        rules: list[GDPRRule],
        tokenizer,
        audit_log: AuditLog | None = None,
        severity_privilege: dict[str, float] | None = None,
    ):
        self.rules = [r for r in rules if r.action == "DENY"]
        self.tokenizer = tokenizer
        self.audit_log = audit_log or AuditLog()
        self.severity_privilege = severity_privilege or SEVERITY_PRIVILEGE

    def allocate(self, model, input_ids: torch.Tensor, rmax: int) -> int:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        triggered = [r for r in self.rules if r.matches(text)]
        if not triggered:
            self._log(text, [], [], "none", rmax, rmax)
            return rmax

        worst = min(
            triggered,
            key=lambda r: _SEVERITY_ORDER.index(r.severity)
            if r.severity in _SEVERITY_ORDER else len(_SEVERITY_ORDER)
        )

        frac = self.severity_privilege.get(worst.severity, 0.05)
        g = max(1, int(rmax * frac))

        categories = [r.category for r in triggered]
        articles   = sorted({r.article for r in triggered if r.article})
        self._log(text, categories, articles, worst.severity, g, rmax)

        arts_str = f"Art.{articles}" if articles else ""
        print(
            f"  [GDPR DENY] {worst.severity.upper()} {arts_str}  "
            f"→ {categories}  →  g={g}/{rmax} ({frac*100:.0f}%)"
        )
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
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
        return output, g

    def _log(
        self,
        text: str,
        categories: list[str],
        articles: list[int],
        severity: str,
        g: int,
        rmax: int,
    ) -> None:
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()
        self.audit_log.record(AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            prompt_hash=prompt_hash,
            triggered_categories=categories,
            articles=articles,
            severity=severity,
            privilege_granted=g,
            rmax=rmax,
        ))


def verify_audit_log(path: str | Path, hmac_key: bytes) -> dict:
    """Verify the integrity of an HMAC-signed audit log.

    Returns:
        {"total": int, "valid": int, "tampered": int, "unsigned": int}
    """
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

            expected = _hmac.new(hmac_key, json.dumps(row, sort_keys=True).encode(), "sha256").hexdigest()
            if _hmac.compare_digest(stored, expected):
                valid += 1
            else:
                tampered += 1

    return {"total": total, "valid": valid, "tampered": tampered, "unsigned": unsigned}
