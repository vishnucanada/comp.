"""
Policy-targeted NLPN training.

Two-objective loss (Kendall et al., 2018):
  Anchor (g=rmax):  CE on all pairs, original targets  → preserve full capability
  Variant (g=low):  CE on permitted pairs, original targets → preserve allowed
                    CE on denied pairs, refusal targets    → suppress denied
"""
from __future__ import annotations
import random
import torch
import torch.nn as nn
from .enforcer import set_privilege, get_rmax


def train_nlpn_policy(
    model: nn.Module,
    tokenizer,
    training_pairs: list,
    rmax: int | None = None,
    low_privilege: int | None = None,
    n_steps: int = 200,
    batch_size: int = 4,
    lr: float = 5e-5,
    log_every: int = 20,
    device: str = "cpu",
    max_length: int = 128,
) -> dict[str, list[float]]:
    if rmax is None:
        rmax = get_rmax(model)
    if low_privilege is None:
        low_privilege = max(1, rmax // 10)

    log_vars = {
        "anchor":  nn.Parameter(torch.zeros(1, device=device)),
        "variant": nn.Parameter(torch.zeros(1, device=device)),
    }
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(log_vars.values()), lr=lr
    )

    history: dict[str, list[float]] = {
        "total": [], "anchor": [], "variant_permitted": [], "variant_denied": []
    }
    rng = random.Random(0)

    for step in range(n_steps):
        batch     = rng.choices(training_pairs, k=batch_size)
        permitted = [p for p in batch if p.split == "permitted"]
        denied    = [p for p in batch if p.split == "denied"]

        set_privilege(model, rmax)
        loss_anchor = _batch_loss(model, tokenizer, batch, batch, max_length, device)

        set_privilege(model, low_privilege)
        loss_permitted = (
            _batch_loss(model, tokenizer, permitted, permitted, max_length, device)
            if permitted else torch.zeros(1, device=device)
        )
        if denied:
            refusal_text    = denied[0].completion
            refusal_targets = [
                type(p)(prompt=p.prompt, completion=refusal_text,
                        split=p.split, category=p.category)
                for p in denied
            ]
            loss_denied = _batch_loss(model, tokenizer, denied, refusal_targets, max_length, device)
        else:
            loss_denied = torch.zeros(1, device=device)

        s_a, s_v = log_vars["anchor"], log_vars["variant"]
        total = (
            torch.exp(-s_a) * loss_anchor + s_a +
            torch.exp(-s_v) * (loss_permitted + loss_denied) / 2 + s_v
        )

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history["total"].append(total.item())
        history["anchor"].append(loss_anchor.item())
        history["variant_permitted"].append(loss_permitted.item())
        history["variant_denied"].append(loss_denied.item())

        if (step + 1) % log_every == 0:
            print(
                f"step {step+1:>{len(str(n_steps))}}/{n_steps}  "
                f"total={total.item():.3f}  "
                f"anchor={loss_anchor.item():.3f}  "
                f"denied={loss_denied.item():.3f}  "
                f"permitted={loss_permitted.item():.3f}"
            )

    set_privilege(model, rmax)
    return history


def _batch_loss(model, tokenizer, input_pairs, target_pairs, max_length, device):
    if not input_pairs:
        return torch.zeros(1, device=device, requires_grad=True)
    ids, labels = zip(*[
        _tokenize_pair(tokenizer, inp.prompt, tgt.completion, max_length)
        for inp, tgt in zip(input_pairs, target_pairs)
    ])
    return model(
        input_ids=torch.stack(ids).to(device),
        labels=torch.stack(labels).to(device),
    ).loss


def _tokenize_pair(tokenizer, prompt: str, completion: str, max_length: int):
    p_ids = tokenizer.encode(prompt, add_special_tokens=True)
    c_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos   = tokenizer.eos_token_id
    pad   = tokenizer.pad_token_id or eos

    inp = (p_ids + c_ids + [eos])[:max_length]
    lbl = ([-100] * len(p_ids) + c_ids + [eos])[:max_length]
    n   = max_length - len(inp)

    return (
        torch.tensor(inp + [pad] * n, dtype=torch.long),
        torch.tensor(lbl + [-100] * n, dtype=torch.long),
    )
