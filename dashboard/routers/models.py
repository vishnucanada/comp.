"""LLM-backend discovery for the dashboard chat panel."""

import os

from fastapi import APIRouter

from ..backends import ollama_model
from ..config import ANTHROPIC_MODEL

router = APIRouter()


@router.get("/api/models")
async def list_models():
    """Report which chat backends are reachable so the UI can show a status."""
    backends = []
    model = ollama_model()
    if model:
        backends.append({"type": "ollama", "active": model})
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append({"type": "anthropic", "active": ANTHROPIC_MODEL})
    return {"backends": backends, "ready": bool(backends)}
