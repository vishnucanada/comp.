"""Admin routes: audit log and policy library."""

import re

from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import ROOT
from ..deps import _require_auth
from ..helpers import _persist_policy, get_admin_log, log_admin_action

router = APIRouter(prefix="/api/admin")

_LIBRARY_DIR = ROOT / "policies" / "library"


@router.get("/audit-log")
async def audit_log(limit: int = 100, _auth=Depends(_require_auth)):
    return {"entries": get_admin_log(limit=limit)}


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
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", policy_id)
    f = _LIBRARY_DIR / f"{safe}.txt"
    if not f.exists():
        raise HTTPException(404, "Library policy not found")
    return {"id": safe, "content": f.read_text()}


@router.post("/policy-library/{policy_id}/install")
async def install_library_policy(policy_id: str, request: Request, _auth=Depends(_require_auth)):
    """Copy a library policy into the tenant's active policies directory."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", policy_id)
    f = _LIBRARY_DIR / f"{safe}.txt"
    if not f.exists():
        raise HTTPException(404, "Library policy not found")

    content = f.read_text()
    first_line = content.split("\n")[0]
    description = (
        first_line.replace("name:", "").strip() if first_line.startswith("name:") else safe
    )
    _persist_policy(safe, description, content)

    log_admin_action(request, "install_library_policy", safe)
    return {"installed": True, "name": safe}
