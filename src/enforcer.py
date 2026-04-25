"""Enforcement operator: wrap a model with NLPN layers and control privilege."""
import torch.nn as nn
from .nlpn import NLPNLinear

# Default MLP projection names across common architectures.
# Llama/Qwen (SwiGLU), GPT-2, Pythia/GPT-NeoX, generic BERT-style.
_DEFAULT_TARGET_MODULES = [
    "down_proj", "gate_proj", "up_proj",  # Llama, Qwen, Mistral
    "c_fc", "c_proj",                      # GPT-2
    "dense_h_to_4h", "dense_4h_to_h",     # Pythia, GPT-NeoX
    "fc1", "fc2",                          # BERT, generic
]


def wrap_with_nlpn(
    model: nn.Module,
    rmax: int,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Replace matching nn.Linear layers with NLPNLinear layers (the enforcer Tg).

    Args:
        model: Pretrained transformer.
        rmax: Maximum rank (= full privilege ceiling).
        target_modules: Module name substrings to replace.
            Defaults to common MLP projection names.

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
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, NLPNLinear.from_linear(module, rmax))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            f"No matching nn.Linear layers found. "
            f"Searched for: {targets}. "
            f"Pass target_modules= with the correct layer names for your architecture."
        )
    print(f"Wrapped {replaced} linear layers with NLPNLinear (rmax={rmax})")
    return model


def set_privilege(model: nn.Module, g: int) -> None:
    """Set privilege level g on every NLPNLinear in the model."""
    for module in model.modules():
        if isinstance(module, NLPNLinear):
            module.privilege = g


def get_rmax(model: nn.Module) -> int:
    """Return rmax from the first NLPNLinear found."""
    for module in model.modules():
        if isinstance(module, NLPNLinear):
            return module.rmax
    raise ValueError("No NLPNLinear layers found in model.")


def detect_rmax(
    model: nn.Module,
    target_modules: list[str] | None = None,
) -> int:
    """
    Return the appropriate rmax for this model: min(dout, din) of the first
    matching target layer. This ensures W(rmax) ≈ W_original (full SVD recovery).

    rmax = min dimension because SVD of W ∈ R^(dout×din) has at most
    min(dout, din) non-zero singular values.
    """
    targets = target_modules if target_modules is not None else _DEFAULT_TARGET_MODULES
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            return min(module.in_features, module.out_features)
    raise RuntimeError(f"No target layers found. Searched for: {targets}")


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
