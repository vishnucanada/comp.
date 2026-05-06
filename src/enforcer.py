"""Enforcement operator: wrap a model with NLPN layers and control privilege."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .nlpn import NLPNLinear

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
    """Load tokenizer and model onto the best available device.

    Use load_in_8bit/4bit only for read-only inference on models that will NOT
    be wrapped with wrap_with_nlpn — quantised layers replace nn.Linear before
    wrapping can target them.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
            print("Warning: bitsandbytes not installed — loading in float32 instead.")
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


# MLP and attention projection names across common architectures.
_DEFAULT_TARGET_MODULES = [
    # MLP projections
    "down_proj",
    "gate_proj",
    "up_proj",  # Llama, Qwen, Mistral
    "c_fc",  # GPT-2 MLP
    "dense_h_to_4h",
    "dense_4h_to_h",  # Pythia, GPT-NeoX
    "fc1",
    "fc2",  # BERT, generic
    # Attention projections
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # Llama, Qwen, Mistral
    "c_attn",
    "c_proj",  # GPT-2 (fused QKV + shared output)
    "query_key_value",  # Pythia, GPT-NeoX
    "query",
    "key",
    "value",  # BERT
]


def wrap_with_nlpn(
    model: nn.Module,
    rmax: int | None = None,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Replace matching nn.Linear layers with NLPNLinear layers (the enforcer Tg).

    Each layer's rmax is min(out_features, in_features) — capped by the optional
    `rmax` argument if supplied.  This ensures W(rmax) == W_original for every layer
    independently, regardless of dimension differences across the network.

    Args:
        model: Pretrained transformer.
        rmax: Optional upper bound on any layer's rmax.  If None, each layer
              uses its own min(out, in).
        target_modules: Module name substrings to replace.
            Defaults to MLP + attention projections for common architectures.

    Returns:
        The model with NLPN layers in-place (same object).
    """
    targets = target_modules if target_modules is not None else _DEFAULT_TARGET_MODULES
    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in targets):
            continue
        layer_rmax = min(module.out_features, module.in_features)
        if rmax is not None:
            layer_rmax = min(layer_rmax, rmax)
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, NLPNLinear.from_linear(module, layer_rmax))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            f"No matching nn.Linear layers found. "
            f"Searched for: {targets}. "
            f"Pass target_modules= with the correct layer names for your architecture."
        )
    print(f"Wrapped {replaced} linear layers with NLPNLinear (per-layer rmax)")
    return model


def set_privilege(model: nn.Module, g: int | dict[str, int]) -> None:
    """Set privilege level on NLPNLinear layers.

    g is clamped to each layer's own rmax so that a single global value can be
    applied across layers that have different dimensions (and therefore different
    rmax values).  Layers with rmax < g run at full privilege for their size.

    Args:
        g: A single int applied to all layers, **or** a dict mapping layer-name
           substrings to privilege levels.  Exact name matches take priority;
           unmatched layers keep their current privilege.

    Examples::

        set_privilege(model, 4)                          # all layers → min(4, layer.rmax)
        set_privilege(model, {"q_proj": 2, "down_proj": 6})  # per-component
    """
    if isinstance(g, int):
        for module in model.modules():
            if isinstance(module, NLPNLinear):
                module.privilege = min(g, module.rmax)
        return

    for name, module in model.named_modules():
        if not isinstance(module, NLPNLinear):
            continue
        # Exact match first, then substring
        new_g = g.get(name)
        if new_g is None:
            for key, val in g.items():
                if key in name:
                    new_g = val
                    break
        if new_g is not None:
            module.privilege = min(new_g, module.rmax)


def get_rmax(model: nn.Module) -> int:
    """Return the maximum rmax across all NLPNLinear layers.

    This is the ceiling of the privilege scale — setting privilege to this value
    puts every layer at full capability (each layer clamps to its own rmax).
    """
    rmaxes = [m.rmax for m in model.modules() if isinstance(m, NLPNLinear)]
    if not rmaxes:
        raise ValueError("No NLPNLinear layers found in model.")
    return max(rmaxes)


def detect_rmax(
    model: nn.Module,
    target_modules: list[str] | None = None,
) -> int:
    """
    Return the privilege scale ceiling for this model before wrapping.

    This is the maximum of min(dout, din) across all matching target layers —
    the same value get_rmax() will return after wrap_with_nlpn() is called.
    Pass this to calibrate_privilege / train_nlpn as the rmax argument.
    """
    targets = target_modules if target_modules is not None else _DEFAULT_TARGET_MODULES
    rmaxes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            rmaxes.append(min(module.in_features, module.out_features))
    if not rmaxes:
        raise RuntimeError(f"No target layers found. Searched for: {targets}")
    return max(rmaxes)


def save_nlpn(
    model: nn.Module,
    path: str | Path,
    model_id: str | None = None,
    low_g: int | None = None,
) -> None:
    """
    Save NLPN layer weights and config to a directory.

    Only the factored A/B matrices are saved — not the full base model.
    To restore: load the same base model, call wrap_with_nlpn(), then load_nlpn().

    Args:
        model:    Model already wrapped with wrap_with_nlpn().
        path:     Directory to write nlpn_weights.pt and nlpn_config.json into.
        model_id: HuggingFace model ID to record in config (e.g. "Qwen/Qwen2.5-0.5B").
        low_g:    Calibrated minimum privilege level; stored in config for inference.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    layer_meta: dict[str, dict] = {}
    weights: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, NLPNLinear):
            layer_meta[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "rmax": module.rmax,
                "bias": module.bias is not None,
            }
            weights[f"{name}.A"] = module.A.data.cpu()
            weights[f"{name}.B"] = module.B.data.cpu()
            if module.bias is not None:
                weights[f"{name}.bias"] = module.bias.data.cpu()

    if not layer_meta:
        raise ValueError("No NLPNLinear layers found — call wrap_with_nlpn() first.")

    cfg: dict = {
        "model_id": model_id,
        "rmax": get_rmax(model),
        "n_layers": len(layer_meta),
        "layers": layer_meta,
    }
    if low_g is not None:
        cfg["low_g"] = low_g

    torch.save(weights, path / "nlpn_weights.pt")
    (path / "nlpn_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved {len(layer_meta)} NLPN layers → {path}")


def load_nlpn(model: nn.Module, path: str | Path) -> nn.Module:
    """
    Load saved NLPN weights into a wrapped model.

    The model must already be wrapped with wrap_with_nlpn() using the same
    rmax and target_modules as when save_nlpn() was called.

    Args:
        model: Model already wrapped with wrap_with_nlpn().
        path:  Directory written by save_nlpn().

    Returns:
        The model with weights restored in-place (same object).
    """
    path = Path(path)
    config = json.loads((path / "nlpn_config.json").read_text())
    weights = torch.load(path / "nlpn_weights.pt", map_location="cpu", weights_only=True)

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, NLPNLinear) and name in config["layers"]:
            module.A.data.copy_(weights[f"{name}.A"])
            module.B.data.copy_(weights[f"{name}.B"])
            if module.bias is not None and f"{name}.bias" in weights:
                module.bias.data.copy_(weights[f"{name}.bias"])
            loaded += 1

    if loaded == 0:
        raise ValueError(
            "No NLPNLinear layers matched the saved config. "
            "Ensure the model is wrapped with the same rmax and target_modules."
        )
    print(f"Loaded {loaded} NLPN layers ← {path}")
    return model



def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
