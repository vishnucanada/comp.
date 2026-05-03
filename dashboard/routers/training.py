"""Training job routes: start, status, and live SSE progress stream."""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import _require_auth
from ..helpers import _load_policy, _safe_name
from ..registry import training_registry
from ..schemas import TrainRequest

router = APIRouter()


@router.post("/api/train/{name}")
async def start_training(name: str, req: TrainRequest, _auth=Depends(_require_auth)):
    safe = _safe_name(name)
    if _load_policy(safe) is None:
        raise HTTPException(404, f"Policy '{safe}' not found")
    training_registry.train_async(
        safe,
        {
            "model_id": req.model_id,
            "epochs": req.epochs,
            "lr": req.lr,
            "orth_reg": req.orth_reg,
            "adversarial": req.adversarial,
        },
    )
    return {"status": "training", "name": safe}


@router.get("/api/train/status")
async def training_status():
    return {"jobs": training_registry.all_status()}


@router.get("/api/train/{name}/stream")
async def training_stream(name: str):
    safe = _safe_name(name)
    status = training_registry.status(safe)

    if status == "not_started":
        raise HTTPException(404, f"No training job for '{safe}'")

    async def event_generator():
        if status in ("done",) or status.startswith("error:"):
            kind = "done" if status == "done" else "error"
            yield f"data: {json.dumps({'type': kind, 'message': status})}\n\n"
            return
        q = training_registry.stream_queue(safe)
        if q is None:
            return
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
