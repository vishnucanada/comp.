"""Shared model-loading utilities."""
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
    *,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: torch.dtype | None = None,
):
    """Load tokenizer and model.

    Args:
        model_id:     HuggingFace model ID or local path.
        device:       Target device.  Defaults to best available (MPS / CUDA / CPU).
        load_in_8bit: Use bitsandbytes 8-bit quantisation (requires ``bitsandbytes``).
        load_in_4bit: Use bitsandbytes 4-bit quantisation (requires ``bitsandbytes``).
        torch_dtype:  Override weight dtype (e.g. ``torch.float16``).
                      Defaults to ``float32`` for CPU/MPS and ``float16`` for CUDA
                      when quantisation is not requested.

    Note on quantisation and NLPNLinear
    ------------------------------------
    ``wrap_with_nlpn`` targets ``nn.Linear`` modules.  When *load_in_4bit* or
    *load_in_8bit* is used, bitsandbytes replaces those linears with its own
    quantised types **before** wrapping, so ``wrap_with_nlpn`` will find nothing
    to replace.  The correct workflow is:

        1. Load in full precision (default).
        2. Call ``wrap_with_nlpn``.
        3. Train / load checkpoint.
        4. Optionally quantise the *non-NLPN* parts yourself.

    Use ``load_in_8bit`` / ``load_in_4bit`` only for read-only inference on a
    model that is *not* being wrapped (e.g. a reference model or the Ollama
    backend).
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {}

    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_cfg
            kwargs["device_map"] = "auto"
        except ImportError:
            print(
                "Warning: bitsandbytes not installed — loading in float32 instead.\n"
                "Install with: pip install bitsandbytes"
            )
            load_in_4bit = load_in_8bit = False

    if not (load_in_4bit or load_in_8bit):
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        elif device.type == "cuda":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not (load_in_4bit or load_in_8bit):
        model.to(device)
    model.eval()
    return model, tokenizer
