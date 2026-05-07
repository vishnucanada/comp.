"""Policy enforcement gate — wraps any LLM call or tool invocation.

Enforcement at the call boundary. Works with any backend without model training.

Usage::

    gate = PolicyGate(policy, iam=iam_config, audit_log_path="audit/gate.jsonl")

    response, decision = gate.complete(
        message="What is John's salary?",
        fn=lambda m: client.messages.create(...).content[0].text,
        user_role="anonymous",
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .policy import Policy, PolicyCompiler


@dataclass
class GateDecision:
    allowed: bool
    categories: list[str]
    user_role: str | None = None

    @property
    def denied(self) -> bool:
        return not self.allowed

    def __repr__(self) -> str:
        if self.allowed:
            return f"GateDecision(ALLOW, role={self.user_role!r})"
        return f"GateDecision(DENY{self.categories}, role={self.user_role!r})"


_DENY_MESSAGE = "This request cannot be fulfilled under the current access policy."


class PolicyGate:
    """Enforce a Policy at the API boundary — no model training required.

    Combines three enforcement layers:
      1. Role-based access: IAMConfig roles with privilege="full" bypass content checks.
      2. Tool allowlists: roles may be restricted to a named subset of tools.
      3. Content policy: PolicyCompiler deny-rule matching on the prompt text.

    Every decision can be appended to a JSONL audit log.
    """

    def __init__(
        self,
        policy: Policy,
        iam=None,  # IAMConfig — optional, avoids circular import at module level
        audit_log_path: str | Path | None = None,
        deny_message: str = _DENY_MESSAGE,
    ):
        self.policy = policy
        self.compiler = PolicyCompiler(policy)
        self.iam = iam
        self.deny_message = deny_message
        self._log_path: Path | None = Path(audit_log_path) if audit_log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── core check ───────────────────────────────────────────────────────────

    def check(
        self,
        message: str,
        user_role: str | None = None,
        history: list[str] | None = None,
    ) -> GateDecision:
        """Return a GateDecision for the given message and role."""
        # Full-privilege roles bypass content checks entirely.
        # Use find_role() — not get_role() — so unknown role names cannot silently
        # inherit the default role's privilege and bypass enforcement.
        if self.iam and user_role:
            role = self.iam.find_role(user_role)
            if role is not None and role.privilege == "full":
                decision = GateDecision(allowed=True, categories=[], user_role=user_role)
                self._record(message, decision)
                return decision

        violated, categories = self.compiler.check(message, history)
        decision = GateDecision(allowed=not violated, categories=categories, user_role=user_role)
        self._record(message, decision)
        return decision

    # ── high-level helpers ───────────────────────────────────────────────────

    def complete(
        self,
        message: str,
        fn: Callable[[str], str],
        user_role: str | None = None,
        history: list[str] | None = None,
    ) -> tuple[str, GateDecision]:
        """Gate a completion call. Returns (response, decision).

        If the policy denies the message, fn is never called and deny_message is
        returned as the response.
        """
        decision = self.check(message, user_role=user_role, history=history)
        if decision.denied:
            return self.deny_message, decision
        return fn(message), decision

    def filter_context(
        self,
        documents: list[str],
        user_role: str | None = None,
    ) -> list[str]:
        """Remove documents that would violate policy for this role."""
        return [d for d in documents if self.check(d, user_role=user_role).allowed]

    def run_tool(
        self,
        tool_name: str,
        fn: Callable[..., Any],
        args: dict[str, Any],
        prompt: str = "",
        user_role: str | None = None,
    ) -> tuple[Any, GateDecision]:
        """Gate a tool call: check tool allowlist, then check prompt content.

        Returns (result, decision). result is None if the call was denied.
        """
        if self.iam and user_role:
            role = self.iam.find_role(user_role)
            if role is not None and not role.can_use_tool(tool_name):
                decision = GateDecision(
                    allowed=False,
                    categories=[f"[tool-not-allowed:{tool_name}]"],
                    user_role=user_role,
                )
                self._record(prompt, decision)
                return None, decision

        if prompt:
            decision = self.check(prompt, user_role=user_role)
            if decision.denied:
                return None, decision

        result = fn(**args)
        decision = GateDecision(allowed=True, categories=[], user_role=user_role)
        return result, decision

    # ── audit log ────────────────────────────────────────────────────────────

    def _record(self, message: str, decision: GateDecision) -> None:
        if not self._log_path:
            return
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_hash": hashlib.sha256(message.encode()).hexdigest(),
            "user_role": decision.user_role,
            "allowed": decision.allowed,
            "categories": decision.categories,
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def audit_summary(self) -> str:
        if not self._log_path or not self._log_path.exists():
            return "No audit log."
        rows = []
        with self._log_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        total = len(rows)
        denied = sum(1 for r in rows if not r.get("allowed"))
        cats: dict[str, int] = {}
        for r in rows:
            for c in r.get("categories", []):
                cats[c] = cats.get(c, 0) + 1
        lines = [f"Gate audit: {total} requests, {denied} denied"]
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"  {c:<40} {n}x")
        return "\n".join(lines)
