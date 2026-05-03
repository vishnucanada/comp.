"""
Policy-to-privilege compiler.

Maps a natural-language policy text file to runtime monitors and privilege allocation.

Pipeline:
  policy.txt → PolicyParser → Policy → PolicyCompiler → PolicyAllocator → g

The PolicyAllocator sits in Layer 2 (Allocator) of the MAE stack.
At inference it calls the compiler's check() on the decoded prompt;
if any DENY rule fires, it returns low_privilege instead of rmax.

Without Algorithm 1 training, low privilege produces degraded/garbage output.
With training (train_nlpn), the model learns that at low privilege the denied
capability is genuinely unreachable — not just output-filtered.

Policy file format
------------------
# Comment
name: Policy Name

DENY: category label
  match: keyword1, keyword2, ...   (case-insensitive substring match)
  regex: pattern                   (compiled as re.search on raw prompt)

ALLOW: category label
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass
class PolicyRule:
    action: Literal["DENY", "ALLOW"]
    category: str
    keywords: list[str] = field(default_factory=list)
    patterns: list[re.Pattern] = field(default_factory=list)
    severity: str = "none"  # GDPR tier: critical | high | medium | none
    article: int | None = None  # GDPR article number

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.keywords):
            return True
        return bool(any(p.search(text) for p in self.patterns))

    def __repr__(self) -> str:
        return f"PolicyRule({self.action}, {self.category!r})"


def _combine(text: str, history: list[str] | None) -> str:
    """Prepend the last 3 history turns to catch multi-turn extraction attacks."""
    if not history:
        return text
    return " ".join(history[-3:]) + " " + text


@dataclass
class Policy:
    name: str
    rules: list[PolicyRule]

    @property
    def denied(self) -> list[PolicyRule]:
        return [r for r in self.rules if r.action == "DENY"]

    @property
    def allowed(self) -> list[PolicyRule]:
        return [r for r in self.rules if r.action == "ALLOW"]

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        return _parse(Path(path).read_text())

    @classmethod
    def from_text(cls, text: str) -> Policy:
        return _parse(text)

    def to_text(self) -> str:
        lines = [f"name: {self.name}", ""]
        for rule in self.rules:
            lines.append(f"{rule.action}: {rule.category}")
            if rule.keywords:
                lines.append(f"  match: {', '.join(rule.keywords)}")
            for pat in rule.patterns:
                lines.append(f"  regex: {pat.pattern}")
            lines.append("")
        return "\n".join(lines).strip()

    def summary(self) -> str:
        lines = [f"Policy: {self.name}"]
        for r in self.rules:
            kw = ", ".join(r.keywords[:3])
            lines.append(f"  {r.action:5} {r.category!r:<30} keywords=[{kw}...]")
        return "\n".join(lines)


def _parse(text: str) -> Policy:
    name = "Unnamed Policy"
    rules: list[PolicyRule] = []
    current: PolicyRule | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip()

        elif line.upper().startswith("DENY:"):
            if current:
                rules.append(current)
            current = PolicyRule("DENY", line.split(":", 1)[1].strip())

        elif line.upper().startswith("ALLOW:"):
            if current:
                rules.append(current)
            current = PolicyRule("ALLOW", line.split(":", 1)[1].strip())

        elif current and line.lower().startswith("match:"):
            kws = [k.strip().lower() for k in line.split(":", 1)[1].split(",")]
            current.keywords.extend(k for k in kws if k)

        elif current and line.lower().startswith("regex:"):
            pattern_str = line.split(":", 1)[1].strip()
            try:
                current.patterns.append(re.compile(pattern_str))
            except re.error as e:
                print(f"Warning: skipping invalid regex {pattern_str!r}: {e}")

        elif current and line.lower().startswith("severity:"):
            current.severity = line.split(":", 1)[1].strip().lower()

        elif current and line.lower().startswith("article:"):
            with contextlib.suppress(ValueError):
                current.article = int(line.split(":", 1)[1].strip())

    if current:
        rules.append(current)

    return Policy(name=name, rules=rules)


class PolicyCompiler:
    """
    Compile a Policy into a callable runtime monitor.

    check(text, history=None) → (violated: bool, categories: list[str])
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def check(
        self,
        text: str,
        history: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        combined = _combine(text, history)
        violated = [r.category for r in self.policy.denied if r.matches(combined)]
        return bool(violated), violated

    def __repr__(self) -> str:
        cats = [r.category for r in self.policy.denied]
        return f"PolicyCompiler(deny={cats})"


class PolicyAllocator:
    """Allocator driven by a compiled policy: DENY → low_privilege, else rmax."""

    def __init__(self, compiler: PolicyCompiler, tokenizer, low_privilege: int = 1):
        self.compiler = compiler
        self.tokenizer = tokenizer
        self.low_privilege = low_privilege

    def allocate(
        self,
        model,
        input_ids: torch.Tensor,
        rmax: int,
        history: list[str] | None = None,
        **_,
    ) -> int:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        violated, categories = self.compiler.check(text, history)
        if violated:
            print(f"  [policy] DENY → {categories}  (privilege: {rmax} → {self.low_privilege})")
        else:
            print(f"  [policy] ALLOW  (privilege: {rmax})")
        return self.low_privilege if violated else rmax

    def generate(
        self, model, input_ids, attention_mask=None, rmax=None, history=None, **generate_kwargs
    ):
        from .enforcer import get_rmax, set_privilege

        if rmax is None:
            rmax = get_rmax(model)
        g = self.allocate(model, input_ids, rmax, history=history)
        set_privilege(model, g)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
        return output, g
