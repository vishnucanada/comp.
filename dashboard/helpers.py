"""Shared helpers: name sanitisation, policy persistence, gate construction, audit log."""

import json
import os
import re
import threading
import time
from functools import lru_cache

from fastapi import HTTPException, Request

from .config import AUDIT_DIR, GUARD_BACKEND, POLICIES_DIR

_ADMIN_LOG_PATH = AUDIT_DIR / "admin_actions.jsonl"
_GATE_LOG_PATH = AUDIT_DIR / "gate.jsonl"
_log_lock = threading.Lock()


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name)


def _safe_name(name: str) -> str:
    safe = sanitize(name)
    if not safe:
        raise HTTPException(400, "Invalid policy name")
    return safe


def _persist_policy(safe: str, description: str, structured: str | None) -> None:
    """Write policy files to disk. Invalidates the gate cache for this name."""
    meta_file = POLICIES_DIR / f"{safe}.json"
    txt_file = POLICIES_DIR / f"{safe}.txt"
    meta_file.write_text(json.dumps({"description": description}))
    if structured:
        txt_file.write_text(structured)
    _gate_for_policy.cache_clear()


def _load_policy(name: str):
    """Load a Policy by name from disk. Returns None if not found."""
    from src.policy import Policy

    safe = sanitize(name)
    if not safe:
        return None
    txt = POLICIES_DIR / f"{safe}.txt"
    if txt.exists():
        return Policy.from_file(txt)
    return None


@lru_cache(maxsize=64)
def _gate_for_policy(safe_name: str):
    """Build (and cache) a PolicyGate for a saved policy.

    Cache is cleared whenever a policy is saved or deleted via _persist_policy.
    """
    from src.gate import PolicyGate
    from src.guard import make_guard

    policy = _load_policy(safe_name)
    if policy is None:
        return None
    try:
        guard = make_guard(policy, backend=GUARD_BACKEND)
    except Exception:
        # Fall back to keyword guard if the configured backend can't initialise
        # (missing API key, missing dep, etc.) — better than 500'ing the chat.
        guard = make_guard(policy, backend="keyword")
    return PolicyGate(policy, guard=guard, audit_log_path=_GATE_LOG_PATH)


def policy_check(
    policy_name: str | None,
    message: str,
    user_role: str | None = None,
    history: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Return (decision, violations) for a message against a named policy."""
    if not policy_name:
        return "ALLOW", []
    safe = sanitize(policy_name)
    if not safe:
        return "ALLOW", []
    gate = _gate_for_policy(safe)
    if gate is None:
        return "ALLOW", []
    decision = gate.check(message, user_role=user_role, history=history)
    return ("DENY" if decision.denied else "ALLOW"), decision.categories


def log_admin_action(
    request: Request,
    action: str,
    resource: str,
    details: dict | None = None,
) -> None:
    """Append an admin action entry to the audit log."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ip": request.client.host if request.client else "unknown",
        "action": action,
        "resource": resource,
        "details": details or {},
    }
    hmac_key_hex = os.environ.get("ADMIN_AUDIT_HMAC_KEY", "")
    if hmac_key_hex:
        import hmac as _hmac

        sig = _hmac.new(
            hmac_key_hex.encode(),
            json.dumps(entry, sort_keys=True).encode(),
            "sha256",
        ).hexdigest()
        entry["_hmac"] = sig

    with _log_lock, _ADMIN_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def get_admin_log(limit: int = 100) -> list[dict]:
    if not _ADMIN_LOG_PATH.exists():
        return []
    try:
        lines = _ADMIN_LOG_PATH.read_text().splitlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
        return list(reversed(entries))
    except Exception:
        return []
