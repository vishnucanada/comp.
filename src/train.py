"""Algorithm 1: NLPN privilege-aligned fine-tuning.

Teaches the model that at low privilege, restricted capabilities are
genuinely unreachable — not output-filtered. At full privilege, normal
behaviour is preserved.

What is trained:
  Only the B matrices (output projections) of every NLPNLinear layer.
  A matrices (input projections) and biases are frozen.

  Why: W(g) = B[:, :g] @ A[:g, :].  The B columns at index < low_g
  exclusively determine low-privilege output.  Freezing A and biases
  prevents the deny signal from polluting full-privilege behaviour.

Compute note:
  For Qwen2.5-0.5B (rmax ≈ 896, ~28 MLP layers × 3 projections each),
  the trainable B matrices total ≈ 84 M parameters — about 1/6 of the
  full model.  Training runs on CPU or Apple Silicon MPS in minutes on
  a small dataset.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW

from .enforcer import set_privilege, get_rmax
from .nlpn import NLPNLinear
from .policy import Policy


_REFUSAL = "I'm sorry, I can't help with that."

_DEFAULT_ALLOW: list[tuple[str, str]] = [
    ("What is 2 + 2?", "4."),
    ("What is the capital of France?", "Paris."),
    ("What does HTML stand for?", "HyperText Markup Language."),
    ("Summarize the water cycle.", "Water evaporates, condenses, and falls as rain."),
    ("What is the speed of light?", "About 299,792 kilometres per second."),
    ("What programming language is Python?", "Python is a high-level interpreted language."),
    ("What is photosynthesis?", "Plants convert sunlight and CO2 into glucose."),
    ("What is a neural network?", "A model of layered mathematical functions inspired by the brain."),
    ("What are office hours?", "Office hours are times when staff are available for questions."),
    ("What is the company refund policy?", "Refunds are processed within 14 business days."),
]


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-4
    max_seq_len: int = 64
    log_every: int = 10


def build_deny_examples(policy: Policy) -> list[tuple[str, str]]:
    """Auto-generate (prompt, refusal) pairs from a policy's deny rules."""
    examples: list[tuple[str, str]] = []
    for rule in policy.denied:
        for kw in rule.keywords[:4]:
            examples += [
                (f"Tell me the {kw} of the employee.", _REFUSAL),
                (f"What is someone's {kw}?", _REFUSAL),
                (f"Can you share the {kw}?", _REFUSAL),
                (f"Give me the {kw} for this person.", _REFUSAL),
            ]
    return examples


def _tokenize_pair(
    prompt: str,
    response: str,
    tokenizer,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (input_ids, labels) with prompt tokens masked in labels (-100)."""
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    if all(l == -100 for l in labels):
        return None  # response was fully truncated
    t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    l = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    return t, l


def train_nlpn(
    model: nn.Module,
    tokenizer,
    policy: Policy,
    config: TrainConfig | None = None,
    deny_examples: list[tuple[str, str]] | None = None,
    allow_examples: list[tuple[str, str]] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
) -> None:
    """Fine-tune NLPN B matrices so low-privilege behaviour is genuinely restricted.

    Args:
        model:          Model already wrapped with NLPNLinear layers.
        tokenizer:      Matching tokenizer.
        policy:         Policy whose DENY rules drive the deny training set.
        config:         Hyperparameters (compute-friendly defaults).
        deny_examples:  Override auto-generated (prompt, refusal) pairs.
        allow_examples: Override the default benign (prompt, response) pairs.
        on_step:        Optional callback(epoch, step, loss) for progress tracking.
    """
    cfg = config or TrainConfig()
    device = next(model.parameters()).device
    rmax = get_rmax(model)
    # Mirror the "high" severity tier: 5 % of rmax (matches SEVERITY_PRIVILEGE)
    low_g = max(1, rmax // 20)

    deny_data = deny_examples or build_deny_examples(policy)
    allow_data = allow_examples or _DEFAULT_ALLOW

    if not deny_data:
        raise ValueError("No deny examples — policy has no DENY rules with keywords.")

    # Freeze A matrices and biases; only train B (output projections)
    nlpn_layers = [m for m in model.modules() if isinstance(m, NLPNLinear)]
    for layer in nlpn_layers:
        layer.A.requires_grad_(False)
        if layer.bias is not None:
            layer.bias.requires_grad_(False)
    params = [layer.B for layer in nlpn_layers]

    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} params  "
          f"({100*n_trainable/n_total:.1f}%)  "
          f"[{len(nlpn_layers)} B matrices]")
    print(f"low_g={low_g}  rmax={rmax}  epochs={cfg.epochs}  lr={cfg.lr}")
    print(f"deny examples: {len(deny_data)}  allow examples: {len(allow_data)}")
    print("=" * 60)

    optimizer = AdamW(params, lr=cfg.lr)
    model.train()
    global_step = 0

    for epoch in range(cfg.epochs):
        pairs = (
            [("deny", ex) for ex in deny_data] +
            [("allow", ex) for ex in allow_data]
        )
        random.shuffle(pairs)

        epoch_loss = 0.0
        n_steps = 0

        for kind, (prompt, response) in pairs:
            g = low_g if kind == "deny" else rmax
            set_privilege(model, g)

            pair = _tokenize_pair(prompt, response, tokenizer, cfg.max_seq_len, device)
            if pair is None:
                continue
            input_ids, labels = pair

            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            global_step += 1

            if on_step:
                on_step(epoch + 1, global_step, loss.item())
            elif global_step % cfg.log_every == 0:
                print(f"  [{kind:5}] step {global_step:4d}  g={g:4d}  loss={loss.item():.4f}")

        avg = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs}  avg_loss={avg:.4f}  steps={n_steps}")

    set_privilege(model, rmax)
    model.eval()
    print("=" * 60)
    print("Training complete.")
