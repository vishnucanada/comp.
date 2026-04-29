"""Multi-tenancy: per-tenant namespacing, usage metering, billing hooks."""
import json
import threading
import time
from pathlib import Path

from fastapi import HTTPException, Request

from .config import ROOT

_TENANTS_DIR = ROOT / "tenants"
_TENANTS_DIR.mkdir(exist_ok=True)

_usage_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tenant resolution
# ---------------------------------------------------------------------------

def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from X-Tenant-ID header. Falls back to 'default'."""
    return request.headers.get("X-Tenant-ID", "default").strip() or "default"


def tenant_dir(tenant_id: str) -> Path:
    safe = _safe_tenant(tenant_id)
    d = _TENANTS_DIR / safe
    d.mkdir(exist_ok=True)
    return d


def _safe_tenant(tenant_id: str) -> str:
    import re
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", tenant_id)[:32]
    if not safe:
        raise HTTPException(400, "Invalid tenant ID")
    return safe


# ---------------------------------------------------------------------------
# Tenant-scoped directory helpers
# ---------------------------------------------------------------------------

def tenant_policies_dir(tenant_id: str) -> Path:
    d = tenant_dir(tenant_id) / "policies"
    d.mkdir(exist_ok=True)
    return d


def tenant_checkpoints_dir(tenant_id: str) -> Path:
    d = tenant_dir(tenant_id) / "checkpoints"
    d.mkdir(exist_ok=True)
    return d


def tenant_audit_dir(tenant_id: str) -> Path:
    d = tenant_dir(tenant_id) / "audit"
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Usage metering
# ---------------------------------------------------------------------------

def _usage_file(tenant_id: str) -> Path:
    return tenant_dir(tenant_id) / "usage.json"


def _load_usage(tenant_id: str) -> dict:
    f = _usage_file(tenant_id)
    if not f.exists():
        return {"requests": 0, "tokens_estimated": 0, "policy_checks": 0, "generations": 0}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {"requests": 0, "tokens_estimated": 0, "policy_checks": 0, "generations": 0}


def record_usage(tenant_id: str, event: str, tokens: int = 0) -> None:
    """Thread-safe usage counter increment. Call from any route handler."""
    with _usage_lock:
        usage = _load_usage(tenant_id)
        usage["requests"] = usage.get("requests", 0) + 1
        usage["tokens_estimated"] = usage.get("tokens_estimated", 0) + tokens
        usage[event] = usage.get(event, 0) + 1
        _usage_file(tenant_id).write_text(json.dumps(usage))


def get_usage(tenant_id: str) -> dict:
    return _load_usage(tenant_id)


# ---------------------------------------------------------------------------
# Billing hook placeholder
# ---------------------------------------------------------------------------

def billing_hook(tenant_id: str, event: str, units: int = 1) -> None:
    """
    Extend this function to push usage events to a billing provider
    (Stripe metered billing, AWS Marketplace, etc.).
    """
    record_usage(tenant_id, event, tokens=units)
