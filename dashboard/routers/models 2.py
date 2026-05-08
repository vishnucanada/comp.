"""Model-registry routes: list backends, load NLPN checkpoints, check status."""

import os

from fastapi import APIRouter, Depends, HTTPException

from ..backends import ollama_model
from ..config import ANTHROPIC_MODEL, CHECKPOINTS_DIR
from ..deps import _require_auth
from ..helpers import _safe_name
from ..registry import model_registry

router = APIRouter()


@router.get("/api/models")
async def list_models():
    backends = []
    model = ollama_model()
    if model:
        backends.append({"type": "ollama", "active": model})
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append({"type": "anthropic", "active": ANTHROPIC_MODEL})
    return {"backends": backends, "ready": bool(backends)}


@router.get("/api/models/status")
async def models_status():
    return {"models": model_registry.all_status()}


@router.post("/api/models/load/{name}")
async def load_model_endpoint(name: str, _auth=Depends(_require_auth)):
    safe = _safe_name(name)
    if not (CHECKPOINTS_DIR / safe).exists():
        raise HTTPException(404, f"No checkpoint found at nlpn_checkpoints/{safe}")
    model_registry.load_async(safe)
    return {"status": "loading", "name": safe}
