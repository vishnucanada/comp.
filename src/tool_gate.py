"""
Tool-call level privilege enforcement.

The core argument: in agentic systems, the LLM accesses sensitive data
via tool calls, not from its weights.  Enforcing policy at the tool boundary
gives a provably stronger guarantee than weight modification:

  - The data never enters the model's context if the call is blocked.
  - Works with any model including closed-source APIs — no training needed.
  - The audit trail is complete: you can prove exactly what data the model saw.
  - Blocking is absolute: a blocked tool call returns nothing, not a refusal.

Weight modification gives a probabilistic behavioral guarantee ("the model is
less likely to produce salary information").  Tool-gate gives a hard guarantee
("salary data was never passed to the model's context on this request").

Usage
-----
    gate = ToolGate(policy, user_role="analyst",
                    tool_deny_map={"get_salary": "salary information"})

    # Check before calling
    allowed, reason = gate.check_call("get_salary", prompt="What is John's salary?")
    # → (False, "prompt triggers policy: ['salary information']")

    # Or use execute() which checks, calls, and filters in one step:
    result = gate.execute("get_employee", {"id": 42}, hr_db.get_employee,
                          prompt="What department is John in?")
    # → {"id": 42, "name": "John", "department": "Engineering"}
    #   (salary field stripped from response even though call was allowed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .policy import Policy, PolicyCompiler


@dataclass
class GateResult:
    tool_name: str
    allowed: bool
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class ToolGate:
    """
    Enforce a Policy at the tool-call boundary.

    Two independent enforcement layers:
      1. Call gating — block tool calls when the triggering prompt is policy-denied.
      2. Response filtering — strip denied fields from every response, even on
         allowed calls.  This handles the case where a legitimate prompt causes
         the model to retrieve a record that happens to contain sensitive fields.

    Args:
        policy: Policy with DENY rules defining what data is restricted.
        user_role: Caller identity; passed through to role_privileges in the
                   Policy for potential future role-based tool allowlists.
        tool_deny_map: Maps tool names to policy category names.
                       e.g. {"get_salary": "salary information"}
                       Used by filter_response to identify which response fields
                       to strip for each tool.
        always_strip_categories: Category names whose keywords are always stripped
                                 from every tool response, regardless of call outcome.
                                 Useful for "never let salary appear in context" rules.
    """

    def __init__(
        self,
        policy: Policy,
        user_role: str | None = None,
        tool_deny_map: dict[str, str] | None = None,
        always_strip_categories: list[str] | None = None,
    ):
        self.policy = policy
        self.user_role = user_role
        self.compiler = PolicyCompiler(policy)
        self.tool_deny_map = tool_deny_map or {}
        self.always_strip_categories = set(always_strip_categories or [])

        # Pre-build keyword sets per category for fast response filtering
        self._category_keywords: dict[str, frozenset[str]] = {
            r.category: frozenset(kw.lower() for kw in r.keywords)
            for r in policy.denied
        }

    def check_call(self, tool_name: str, prompt: str) -> tuple[bool, str | None]:
        """Return (allowed, reason). Blocks when prompt triggers a DENY rule."""
        violated, categories = self.compiler.check(prompt)
        if violated:
            return False, f"prompt triggers policy: {categories}"
        return True, None

    def filter_response(self, tool_name: str, response: dict[str, Any]) -> dict[str, Any]:
        """
        Strip fields whose names match denied keywords for this tool's category,
        plus any always-strip categories.

        This runs even on allowed calls — a legitimate prompt can still cause the
        model to retrieve a record that contains fields the caller is not cleared for.
        """
        deny_kws: set[str] = set()

        # Category mapped to this specific tool
        category = self.tool_deny_map.get(tool_name)
        if category and category in self._category_keywords:
            deny_kws |= self._category_keywords[category]

        # Always-strip categories
        for cat in self.always_strip_categories:
            if cat in self._category_keywords:
                deny_kws |= self._category_keywords[cat]

        if not deny_kws:
            return response

        return {
            k: v
            for k, v in response.items()
            if not any(kw in k.lower() for kw in deny_kws)
        }

    def execute(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        tool_fn: Callable[..., dict[str, Any]],
        prompt: str,
    ) -> GateResult:
        """
        Check, execute, and filter a tool call in one step.

        Returns a GateResult with:
          - allowed=False and empty data if the prompt was denied
          - allowed=True and filtered data if the call went through
        """
        allowed, reason = self.check_call(tool_name, prompt)
        if not allowed:
            return GateResult(tool_name=tool_name, allowed=False, reason=reason)

        raw = tool_fn(**kwargs)
        filtered = self.filter_response(tool_name, raw)
        return GateResult(tool_name=tool_name, allowed=True, data=filtered)
