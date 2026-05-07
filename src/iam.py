"""
IAM (Identity and Access Management) configuration for AI.

Same mental model as cloud IAM: define roles, assign privilege levels and
tool allowlists, load from a config file. Replaces the ad-hoc role_privileges
dict in PolicyAllocator with a first-class, auditable configuration object.

Config format (YAML or dict)
-----------------------------
    default_role: anonymous

    roles:
      anonymous:
        privilege: low          # low | medium | full | "25%" | int
        tools: []               # [] = none, "*" = all, or list of tool names

      employee:
        privilege: medium
        tools: [search_docs, get_faq]

      hr_manager:
        privilege: full
        tools: "*"
        description: "Full access for HR staff"

Privilege levels
----------------
    low     → checkpoint's calibrated low_g  (or max(1, rmax // 20))
    medium  → rmax // 2
    full    → rmax
    "25%"   → max(1, rmax * 25 // 100)
    int     → min(value, rmax)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Role:
    name: str
    privilege: str | int = "low"
    tools: list[str] | str = field(default_factory=list)
    description: str = ""

    def can_use_tool(self, tool_name: str) -> bool:
        if self.tools == "*":
            return True
        if isinstance(self.tools, list):
            return tool_name in self.tools
        return False

    def resolve_privilege(self, rmax: int, low_g: int = 1) -> int:
        p = self.privilege
        if p == "full":
            return rmax
        if p == "medium":
            return max(1, rmax // 2)
        if p == "low":
            return low_g
        if isinstance(p, str) and p.endswith("%"):
            pct = float(p[:-1]) / 100.0
            return max(1, int(rmax * pct))
        if isinstance(p, int):
            return min(p, rmax)
        return low_g


class IAMConfig:
    """
    Role registry mapping identities to privilege levels and tool allowlists.

    Load from a YAML file, a dict, or build programmatically.
    Pass to PolicyAllocator or PolicyGate as the authoritative access policy.
    """

    def __init__(
        self,
        roles: dict[str, Role],
        default_role: str = "anonymous",
    ):
        self.roles = roles
        self.default_role = default_role

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IAMConfig":
        default_role = data.get("default_role", "anonymous")
        roles: dict[str, Role] = {}
        for name, cfg in data.get("roles", {}).items():
            if isinstance(cfg, dict):
                roles[name] = Role(
                    name=name,
                    privilege=cfg.get("privilege", "low"),
                    tools=cfg.get("tools", []),
                    description=cfg.get("description", ""),
                )
            else:
                roles[name] = Role(name=name)
        if default_role not in roles:
            roles[default_role] = Role(name=default_role, privilege="low", tools=[])
        return cls(roles=roles, default_role=default_role)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IAMConfig":
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pip install pyyaml to load IAM config from YAML") from e
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data or {})

    # ── resolution ────────────────────────────────────────────────────────────

    def find_role(self, role_name: str | None) -> Role | None:
        """Return the role if explicitly defined, or None for unknown roles.

        Unlike get_role(), does NOT fall back to default_role — callers must
        handle None explicitly. Use this at security boundaries where silently
        granting default privileges to an unrecognised identity is unsafe.
        """
        if role_name and role_name in self.roles:
            return self.roles[role_name]
        return None

    def get_role(self, role_name: str | None) -> Role:
        """Return the role, falling back to default_role if not found."""
        if role_name and role_name in self.roles:
            return self.roles[role_name]
        return self.roles.get(self.default_role, Role(name=self.default_role))

    def resolve_privilege(self, role_name: str | None, rmax: int, low_g: int = 1) -> int:
        return self.get_role(role_name).resolve_privilege(rmax, low_g)

    def can_use_tool(self, role_name: str | None, tool_name: str) -> bool:
        role = self.find_role(role_name)
        if role is None:
            return False
        return role.can_use_tool(tool_name)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_role": self.default_role,
            "roles": {
                name: {
                    "privilege": role.privilege,
                    "tools": role.tools,
                    **({"description": role.description} if role.description else {}),
                }
                for name, role in self.roles.items()
            },
        }

    def to_yaml(self) -> str:
        try:
            import yaml
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:
            import json
            return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        lines = [f"IAM config  (default: {self.default_role})"]
        for name, role in self.roles.items():
            tools = role.tools if role.tools != "*" else "all"
            lines.append(f"  {name:<20} privilege={role.privilege}  tools={tools}")
        return "\n".join(lines)
