"""Admin routes: RBAC, usage metering, audit log, policy library."""
from fastapi import APIRouter, Depends, Request

from ..config import ROOT
from ..rbac import get_admin_log, log_admin_action, require_admin, require_write_access
from ..tenancy import get_tenant_id, get_usage

router = APIRouter(prefix="/api/admin")

_LIBRARY_DIR = ROOT / "policies" / "library"


@router.get("/audit-log")
async def audit_log(limit: int = 100, _role=Depends(require_admin)):
    return {"entries": get_admin_log(limit=limit)}


@router.get("/usage")
async def usage(request: Request, _role=Depends(require_admin)):
    tenant_id = get_tenant_id(request)
    return {"tenant": tenant_id, "usage": get_usage(tenant_id)}


@router.get("/usage/{tenant_id}")
async def tenant_usage(tenant_id: str, _role=Depends(require_admin)):
    return {"tenant": tenant_id, "usage": get_usage(tenant_id)}


@router.get("/policy-library")
async def policy_library():
    """List all pre-built regulatory policies available in the library."""
    if not _LIBRARY_DIR.exists():
        return {"policies": []}
    policies = []
    for p in sorted(_LIBRARY_DIR.glob("*.txt")):
        first_line = p.read_text().split("\n")[0]
        name = first_line.replace("name:", "").strip() if first_line.startswith("name:") else p.stem
        policies.append({"id": p.stem, "name": name, "file": p.name})
    return {"policies": policies}


@router.get("/policy-library/{policy_id}")
async def get_library_policy(policy_id: str):
    """Return the content of a library policy."""
    import re
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", policy_id)
    f = _LIBRARY_DIR / f"{safe}.txt"
    if not f.exists():
        from fastapi import HTTPException
        raise HTTPException(404, "Library policy not found")
    return {"id": safe, "content": f.read_text()}


@router.post("/policy-library/{policy_id}/install")
async def install_library_policy(
    policy_id: str, request: Request, _role=Depends(require_write_access)
):
    """Copy a library policy into the tenant's active policies directory."""
    import re

    from ..helpers import _persist_policy

    safe = re.sub(r"[^a-zA-Z0-9_-]", "", policy_id)
    f = _LIBRARY_DIR / f"{safe}.txt"
    if not f.exists():
        from fastapi import HTTPException
        raise HTTPException(404, "Library policy not found")

    content = f.read_text()
    first_line = content.split("\n")[0]
    description = first_line.replace("name:", "").strip() if first_line.startswith("name:") else safe
    _persist_policy(safe, description, content)

    log_admin_action(request, "install_library_policy", safe)
    return {"installed": True, "name": safe}
