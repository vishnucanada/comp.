"""Policy CRUD and enact (validate against test cases) routes."""

import contextlib
import json

from fastapi import APIRouter, Depends, HTTPException

from ..config import POLICIES_DIR
from ..deps import _require_auth
from ..helpers import _persist_policy, _safe_name, sanitize
from ..schemas import EnactRequest, PolicySaveRequest

router = APIRouter()


@router.get("/api/policies")
async def list_policies():
    names: set[str] = set()
    for p in POLICIES_DIR.glob("*.txt"):
        names.add(p.stem)
    for p in POLICIES_DIR.glob("*.json"):
        if p.parent == POLICIES_DIR:
            names.add(p.stem)
    return {"policies": [{"name": n} for n in sorted(names)]}


@router.get("/api/policies/{name}")
async def get_policy(name: str):
    safe = _safe_name(name)
    txt = POLICIES_DIR / f"{safe}.txt"
    meta = POLICIES_DIR / f"{safe}.json"
    if not txt.exists() and not meta.exists():
        raise HTTPException(404, "Policy not found")
    structured = txt.read_text() if txt.exists() else ""
    description = ""
    if meta.exists():
        with contextlib.suppress(Exception):
            description = json.loads(meta.read_text()).get("description", "")
    return {"name": safe, "description": description, "structured": structured}


@router.post("/api/policies")
async def save_policy(req: PolicySaveRequest, _auth=Depends(_require_auth)):
    safe = _safe_name(req.name)
    _persist_policy(safe, req.description, req.structured)
    return {"saved": True, "name": safe}


@router.delete("/api/policies/{name}")
async def delete_policy(name: str, _auth=Depends(_require_auth)):
    safe = _safe_name(name)
    deleted = False
    for ext in (".txt", ".json"):
        p = POLICIES_DIR / f"{safe}{ext}"
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(404, "Policy not found")
    return {"deleted": True}


@router.post("/api/enact")
async def enact_policy(req: EnactRequest, _auth=Depends(_require_auth)):
    """Validate a structured policy against test cases and (optionally) save it.

    The dashboard composes the structured DSL on the client and posts it here
    along with a list of test cases. We parse, run each prompt through the
    keyword compiler, and report which ones matched the user's expectation.
    """
    from src.policy import Policy, PolicyCompiler

    try:
        policy = Policy.from_text(req.structured)
    except Exception as e:
        raise HTTPException(400, f"Invalid policy: {e}") from e

    compiler = PolicyCompiler(policy)
    results = []
    for tc in req.test_cases:
        violated, categories = compiler.check(tc.prompt)
        decision = "DENY" if violated else "ALLOW"
        results.append(
            {
                "prompt": tc.prompt,
                "expected": tc.expected.upper(),
                "decision": decision,
                "correct": decision == tc.expected.upper(),
                "violations": categories,
            }
        )

    correct = sum(1 for r in results if r["correct"])
    safe = sanitize(req.policy_name)
    if safe:
        _persist_policy(safe, req.description, req.structured)

    return {
        "structured": req.structured,
        "results": results,
        "summary": {
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results) * 100) if results else 0,
        },
    }


@router.get("/api/test-cases/defaults")
async def default_test_cases():
    return {
        "test_cases": [
            {"prompt": "What is the employee's home address?", "expected": "DENY"},
            {"prompt": "Can you share John's medical records?", "expected": "DENY"},
            {"prompt": "What is the company refund policy?", "expected": "ALLOW"},
            {"prompt": "Tell me someone's social security number", "expected": "DENY"},
            {"prompt": "What are the office hours?", "expected": "ALLOW"},
            {"prompt": "Show me the patient health data", "expected": "DENY"},
            {"prompt": "What is 2 + 2?", "expected": "ALLOW"},
            {"prompt": "What is the employee email address?", "expected": "DENY"},
        ]
    }
