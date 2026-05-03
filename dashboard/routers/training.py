"""Training job routes: start and status."""

from fastapi import APIRouter, Depends, HTTPException

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
        },
    )
    return {"status": "training", "name": safe}


@router.get("/api/train/status")
async def training_status():
    return {"jobs": training_registry.all_status()}
