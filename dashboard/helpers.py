"""Shared helper functions: name sanitisation, policy persistence, policy loading, audit log."""

import json
import os
import re
import threading
import time

from fastapi import HTTPException, Request

from .config import POLICIES_DIR

_ADMIN_LOG_PATH = POLICIES_DIR.parent / "audit" / "admin_actions.jsonl"
_ADMIN_LOG_PATH.parent.mkdir(exist_ok=True)
_log_lock = threading.Lock()


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name)


def _safe_name(name: str) -> str:
    safe = sanitize(name)
    if not safe:
        raise HTTPException(400, "Invalid policy name")
    return safe


def _persist_policy(safe: str, description: str, structured: str | None) -> None:
    """Write policy files to disk."""
    meta_file = POLICIES_DIR / f"{safe}.json"
    txt_file = POLICIES_DIR / f"{safe}.txt"
    meta_file.write_text(json.dumps({"description": description}))
    if structured:
        txt_file.write_text(structured)


def _load_policy(name: str):
    """Load a Policy by name from disk, translating description if no .txt file."""
    from src.policy import Policy
    from src.translator import PolicyTranslator

    safe = sanitize(name)
    if not safe:
        return None
    txt = POLICIES_DIR / f"{safe}.txt"
    meta = POLICIES_DIR / f"{safe}.json"
    if txt.exists():
        return Policy.from_file(txt)
    if meta.exists():
        try:
            desc = json.loads(meta.read_text()).get("description", "")
            if desc:
                return PolicyTranslator().translate(desc)
        except Exception:
            pass
    return None


def policy_check(policy_name: str | None, message: str) -> tuple[str, list[str]]:
    """Return (decision, violations) for a message against a named policy."""
    from src.policy import PolicyCompiler

    if not policy_name:
        return "ALLOW", []
    policy = _load_policy(policy_name)
    if not policy:
        return "ALLOW", []
    violated, violations = PolicyCompiler(policy).check(message)
    return ("DENY" if violated else "ALLOW"), violations


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
