import sys
import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from policy import Policy, PolicyCompiler

app = FastAPI(title="comp. Admin Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

POLICIES_DIR = Path(__file__).parent / "policies"
POLICIES_DIR.mkdir(exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name)


def policy_to_text(policy) -> str:
    lines = [f"name: {policy.name}", ""]
    for rule in policy.rules:
        lines.append(f"{rule.action}: {rule.category}")
        if rule.keywords:
            lines.append(f"  match: {', '.join(rule.keywords)}")
        for pat in rule.patterns:
            lines.append(f"  regex: {pat.pattern}")
        lines.append("")
    return "\n".join(lines).strip()


class PolicySaveRequest(BaseModel):
    name: str
    description: str
    structured: Optional[str] = None


class TestCase(BaseModel):
    prompt: str
    expected: str


class EnactRequest(BaseModel):
    policy_name: str
    description: str
    test_cases: List[TestCase]
    low_privilege: int = 1
    rmax: int = 100


@app.get("/favicon.png")
async def favicon():
    return FileResponse(Path(__file__).parent / "favicon.png", media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse((Path(__file__).parent / "landing.html").read_text())


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse((Path(__file__).parent / "dashboard.html").read_text())


@app.get("/api/policies")
async def list_policies():
    names: set[str] = set()
    for p in POLICIES_DIR.glob("*.txt"):
        names.add(p.stem)
    for p in POLICIES_DIR.glob("*.json"):
        names.add(p.stem)
    return {"policies": [{"name": n} for n in sorted(names)]}


@app.get("/api/policies/{name}")
async def get_policy(name: str):
    txt  = POLICIES_DIR / f"{name}.txt"
    meta = POLICIES_DIR / f"{name}.json"
    if not txt.exists() and not meta.exists():
        raise HTTPException(404, "Policy not found")
    structured  = txt.read_text() if txt.exists() else ""
    description = ""
    if meta.exists():
        try:
            description = json.loads(meta.read_text()).get("description", "")
        except Exception:
            pass
    return {"name": name, "description": description, "structured": structured}


@app.post("/api/policies")
async def save_policy(req: PolicySaveRequest):
    safe = sanitize(req.name)
    if not safe:
        raise HTTPException(400, "Invalid policy name")
    (POLICIES_DIR / f"{safe}.json").write_text(json.dumps({"description": req.description}))
    if req.structured:
        (POLICIES_DIR / f"{safe}.txt").write_text(req.structured)
    return {"saved": True, "name": safe}


@app.delete("/api/policies/{name}")
async def delete_policy(name: str):
    deleted = False
    for ext in (".txt", ".json"):
        p = POLICIES_DIR / f"{name}{ext}"
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(404, "Policy not found")
    return {"deleted": True}


@app.post("/api/enact")
async def enact_policy(req: EnactRequest):
    structured_text = None
    translation_note = None

    # Attempt translation via Claude
    try:
        from translator import PolicyTranslator
        translator = PolicyTranslator()
        policy = translator.translate(req.description)
        structured_text = policy_to_text(policy)
    except ValueError:
        # No API key — fall back to treating description as structured format
        translation_note = "no_api_key"
        try:
            policy = Policy.from_text(req.description)
            structured_text = req.description
        except Exception as e:
            raise HTTPException(400, f"Set ANTHROPIC_API_KEY to enable translation, or write structured policy directly. Error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Translation error: {e}")

    # Run test cases
    compiler = PolicyCompiler(policy)
    results = []
    for tc in req.test_cases:
        violated, categories = compiler.check(tc.prompt)
        privilege = req.low_privilege if violated else req.rmax
        privilege_pct = round((privilege / req.rmax) * 100)
        decision = "DENY" if violated else "ALLOW"
        expected = tc.expected.upper()
        results.append({
            "prompt": tc.prompt,
            "expected": expected,
            "decision": decision,
            "correct": decision == expected,
            "violations": categories,
            "privilege": privilege,
            "privilege_pct": privilege_pct,
        })

    correct = sum(1 for r in results if r["correct"])

    # Auto-save
    safe = sanitize(req.policy_name)
    if safe:
        (POLICIES_DIR / f"{safe}.json").write_text(json.dumps({"description": req.description}))
        if structured_text:
            (POLICIES_DIR / f"{safe}.txt").write_text(structured_text)

    return {
        "structured": structured_text,
        "translation_note": translation_note,
        "results": results,
        "summary": {
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results) * 100) if results else 0,
        },
    }


@app.get("/api/test-cases/defaults")
async def default_test_cases():
    return {"test_cases": [
        {"prompt": "What is the employee's home address?",     "expected": "DENY"},
        {"prompt": "Can you share John's medical records?",    "expected": "DENY"},
        {"prompt": "What is the company refund policy?",       "expected": "ALLOW"},
        {"prompt": "Tell me someone's social security number", "expected": "DENY"},
        {"prompt": "What are the office hours?",               "expected": "ALLOW"},
        {"prompt": "Show me the patient health data",          "expected": "DENY"},
        {"prompt": "What is 2 + 2?",                          "expected": "ALLOW"},
        {"prompt": "What is the employee email address?",      "expected": "DENY"},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
