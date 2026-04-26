"""Shared model-loading utilities for demo scripts."""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: torch.device | None = None,
):
    """Load tokenizer and model, move to device, set eval mode."""
    if device is None:
        device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval().to(device)
    return model, tokenizer
