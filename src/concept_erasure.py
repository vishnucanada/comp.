"""
Linear concept erasure via mean-difference projection.

The argument for why this is stronger than behavioral fine-tuning (NLPN):

  Behavioral training (what NLPN does):
    The model's *outputs* are conditioned to refuse at low privilege.
    But the concept is still encoded in intermediate activations — a linear
    probe trained on those activations can still recover it.  The model
    "knows" the salary but has learned not to say it.

  Concept erasure:
    The concept direction is identified in activation space and projected out
    of every forward pass.  After erasure, a probe cannot recover the concept
    because the relevant information is gone from the representations, not
    just suppressed in the output layer.

This is a simplified version of LEACE (Least-squares Concept Erasure,
Belrose et al. 2023).  The erasure direction is the first principal component
of the mean difference between concept-present and concept-absent activations.

Usage
-----
    eraser = ConceptEraser()
    eraser.fit(salary_activations, non_salary_activations)

    # Hook into the model's forward pass at any layer:
    def hook(module, input, output):
        return eraser.erase(output)
    model.layer.register_forward_hook(hook)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConceptEraser:
    """
    Identify and remove a linear concept direction from activations.

    fit(positive, negative) finds the direction that best separates
    concept-present from concept-absent representations.

    erase(x) projects out that direction: x ← x - (x·d)d
    After erasure, any linear function of x that depended on the concept
    direction now has zero signal.
    """

    def __init__(self):
        self.direction: torch.Tensor | None = None
        self._mean_diff: torch.Tensor | None = None

    def fit(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> "ConceptEraser":
        """
        Compute the erasure direction from labeled activation sets.

        Args:
            positive: (n, d) activations where the concept is present.
            negative: (n, d) activations where the concept is absent.
        """
        mu_pos = positive.float().mean(0)
        mu_neg = negative.float().mean(0)
        diff = mu_pos - mu_neg
        self._mean_diff = diff
        self.direction = diff / (diff.norm() + 1e-8)
        return self

    def erase(self, x: torch.Tensor) -> torch.Tensor:
        """Project out the concept direction from x."""
        if self.direction is None:
            raise RuntimeError("Call fit() before erase().")
        d = self.direction.to(x.device, dtype=x.dtype)
        projection = (x @ d).unsqueeze(-1) * d
        return x - projection

    def concept_score(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar projection of each sample onto the concept direction."""
        if self.direction is None:
            raise RuntimeError("Call fit() before concept_score().")
        d = self.direction.to(x.device, dtype=x.dtype)
        return x @ d

    def hook(self):
        """Return a forward hook that erases the concept from layer outputs."""
        eraser = self

        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return eraser.erase(output)
            # Handle tuple outputs (e.g. attention layers)
            if isinstance(output, tuple):
                return (eraser.erase(output[0]),) + output[1:]
            return output

        return _hook


class LinearProbe(nn.Module):
    """
    Binary linear classifier on activations.  Used to measure whether a
    concept is still decodable from representations after erasure or training.

    If probe accuracy stays high after intervention → concept still in activations.
    If probe accuracy drops to ~50% → concept successfully removed.
    """

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)

    def fit(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.05,
    ) -> "LinearProbe":
        X = torch.cat([positive, negative]).float()
        y = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy(self(X), y)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def accuracy(self, positive: torch.Tensor, negative: torch.Tensor) -> float:
        X = torch.cat([positive, negative]).float()
        y = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
        preds = (self(X) > 0.5).float()
        return (preds == y).float().mean().item()
