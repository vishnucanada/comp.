"""Nested Least-Privilege Linear layer (Section 4 of the paper)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLPNLinear(nn.Module):
    """
    Reparameterizes W ∈ R^(dout×din) as B @ A where
      A ∈ R^(rmax×din),  B ∈ R^(dout×rmax).

    At privilege g, the effective weight is:
      W(g) = B[:, :g] @ A[:g, :]   (rank ≤ g prefix)

    This gives a monotone nested family:
      Im(W(g)) ⊆ Im(W(g+1)) ⊆ ... ⊆ Im(W(rmax))
    """

    def __init__(self, in_features: int, out_features: int, rmax: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rmax = rmax

        self.A = nn.Parameter(torch.empty(rmax, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rmax))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._privilege = rmax

        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rmax: int) -> "NLPNLinear":
        """
        Initialize from a pretrained nn.Linear via SVD.

        Factorizes W ≈ (U * sqrt(S)) @ (sqrt(S) * Vh) so that
        the top-rmax singular components are captured in B[:, :r] @ A[:r, :].
        """
        W = linear.weight.data.float()
        r = min(rmax, W.shape[0], W.shape[1])

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S_sqrt = S[:r].sqrt()
        B_init = U[:, :r] * S_sqrt.unsqueeze(0)   # (dout, r)
        A_init = S_sqrt.unsqueeze(1) * Vh[:r, :]  # (r, din)

        layer = cls(linear.in_features, linear.out_features, rmax,
                    bias=linear.bias is not None)
        layer.B.data.zero_()
        layer.A.data.zero_()

        dtype = linear.weight.dtype
        layer.B.data[:, :r] = B_init.to(dtype)
        layer.A.data[:r, :] = A_init.to(dtype)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    @property
    def privilege(self) -> int:
        return self._privilege

    @privilege.setter
    def privilege(self, g: int) -> None:
        if not (1 <= g <= self.rmax):
            raise ValueError(f"Privilege must be in [1, {self.rmax}], got {g}")
        self._privilege = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._privilege
        W_g = self.B[:, :g] @ self.A[:g, :]
        return F.linear(x, W_g, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rmax={self.rmax}, privilege={self._privilege}")
