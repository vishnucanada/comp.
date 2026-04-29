"""Role-based access control and admin audit trail."""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import HTTPException, Request

from .config import ROOT

_ADMIN_LOG = ROOT / "audit" / "admin_actions.jsonl"
_ADMIN_LOG.parent.mkdir(exist_ok=True)
_log_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

ROLES = {"admin", "policy-editor", "viewer"}

# Route-level required roles (path_prefix → minimum required role set)
_WRITE_ROLES  = {"admin", "policy-editor"}
_ADMIN_ROLES  = {"admin"}

# Role hierarchy: admin > policy-editor > viewer
_ROLE_RANK = {"admin": 3, "policy-editor": 2, "viewer": 1}

# API key → role mapping loaded from RBAC_KEYS env var (JSON string)
# Example: RBAC_KEYS='{"key-abc":"admin","key-xyz":"policy-editor","key-123":"viewer"}'
def _load_key_map() -> dict[str, str]:
    raw = os.environ.get("RBAC_KEYS", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def get_role(request: Request) -> str | None:
    """Resolve role from X-API-Key header. Returns None if key absent/unmapped."""
    key = request.headers.get("X-API-Key", "")
    if not key:
        return None
    return _load_key_map().get(key)


def require_role(allowed: set[str]):
    """FastAPI dependency factory. Usage: Depends(require_role({'admin'}))"""
    def dependency(request: Request):
        from .config import API_KEY
        # Legacy single-key bypass: if API_KEY is set and matches, grant admin
        if API_KEY and request.headers.get("X-API-Key", "") == API_KEY:
            return "admin"
        role = get_role(request)
        if role is None:
            raise HTTPException(401, "Authentication required")
        if role not in allowed:
            raise HTTPException(403, f"Role '{role}' lacks permission; need one of {sorted(allowed)}")
        return role
    return dependency


# Convenience pre-built dependencies
require_admin         = require_role(_ADMIN_ROLES)
require_write_access  = require_role(_WRITE_ROLES)


# ---------------------------------------------------------------------------
# Admin audit trail
# ---------------------------------------------------------------------------

def log_admin_action(
    request: Request,
    action: str,
    resource: str,
    details: dict | None = None,
) -> None:
    """Append a tamper-evident admin action entry to the audit log."""
    role = get_role(request) or "legacy-key"
    entry = {
        "ts":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ip":       request.client.host if request.client else "unknown",
        "role":     role,
        "action":   action,
        "resource": resource,
        "details":  details or {},
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

    with _log_lock:
        with _ADMIN_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def get_admin_log(limit: int = 100) -> list[dict]:
    if not _ADMIN_LOG.exists():
        return []
    try:
        lines = _ADMIN_LOG.read_text().splitlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
        return list(reversed(entries))
    except Exception:
        return []
