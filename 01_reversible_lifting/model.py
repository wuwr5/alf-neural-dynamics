"""Component 1 -- reversible (near-invertible) lifting for longitudinal features.

WHAT THIS IS
------------
IS-FNO (arXiv:2512.19439) forces a *near-reversible pairing* between its lifting
map L0 and projection map L0^dagger, using a reversible residual network. The
motivation in the paper is physical (the inverse scattering transform is an
exactly reversible map between solution space and scattering space), but the
*statistical* consequence is what we care about here:

    an encoder that is forced to be invertible cannot silently discard
    information about its input.

That is a strong, architecture-level prior. With 2,316 patients and 781 events
at the Day-2 landmark, a free-form encoder will happily collapse the trajectory
onto whatever correlates with the label in the training fold; a constrained one
has to keep enough of the signal to rebuild the labs. On small clinical cohorts
this behaves like a powerful regulariser, and it gives you something a normal
encoder does not: a **self-supervised pretext task on unlabelled trajectories**.
This repo has 907,046 longitudinal rows to pretrain on, and only 2,316 labelled
landmark patients -- that asymmetry is exactly what two-stage training exploits.

WHAT THIS IS NOT
----------------
It is not a generative model and it does not assume the clinical course is
reversible. Death is an absorbing state; the reversible constraint lives only on
the *feature lifting*, i.e. on the map from (value, missingness) channels to the
latent channel stack, applied pointwise at each observed time step. Temporal
aggregation afterwards is deliberately NOT invertible.

THE BLOCK
---------
With the input split along the channel axis into (a, b):

    a' = a + g(b + f(a))
    b' = b + f(a)

The inverse is closed-form, no fixed-point iteration:

    a  = a' - g(b')         because b + f(a) = b'
    b  = b' - f(a)

Cost of the inverse = cost of the forward pass, so the reconstruction objective
is cheap enough to run every step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
class ZeroPadLifting(nn.Module):
    """IS-FNO style lifting: identity on real channels, zero-pad the rest.

    Exact inverse = slice. This is the one place where the paper's choice is
    directly better than a learned linear lift: a learned lift is invertible
    only if you constrain it, a zero pad is invertible by construction.
    """

    def __init__(self, d_in: int, d_eps: int) -> None:
        super().__init__()
        if d_eps < d_in:
            raise ValueError(f"d_eps ({d_eps}) must be >= d_in ({d_in})")
        self.d_in, self.d_eps = d_in, d_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.d_eps - self.d_in
        if pad == 0:
            return x
        return F.pad(x, (0, pad))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.d_in]


class _SubMap(nn.Module):
    """One of f / g inside the invertible block: LayerNorm -> Linear -> act -> Linear."""

    def __init__(self, d_in: int, d_out: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )
        # Start as (near) identity so the block is stable at initialisation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InvertibleBlock(nn.Module):
    """Reversible residual block with a closed-form inverse (see module docstring)."""

    def __init__(self, d_a: int, d_b: int, hidden: int) -> None:
        super().__init__()
        self.f = _SubMap(d_a, d_b, hidden)   # a -> b
        self.g = _SubMap(d_b, d_a, hidden)   # b -> a
        self.d_a, self.d_b = d_a, d_b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fa = self.f(a)
        b_prime = b + fa
        a_prime = a + self.g(b_prime)
        return a_prime, b_prime

    def inverse(self, a_prime: torch.Tensor, b_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = a_prime - self.g(b_prime)
        b = b_prime - self.f(a)
        return a, b


class ReversibleStack(nn.Module):
    """Stack of invertible blocks acting on the last (channel) dimension."""

    def __init__(self, d_eps: int, n_blocks: int, hidden: int) -> None:
        super().__init__()
        d_a = d_eps // 2
        d_b = d_eps - d_a
        if d_a == 0 or d_b == 0:
            raise ValueError("d_eps must be >= 2")
        self.d_a = d_a
        self.blocks: nn.ModuleList = nn.ModuleList(
            [InvertibleBlock(d_a, d_b, hidden) for _ in range(n_blocks)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a, b = z[..., : self.d_a], z[..., self.d_a:]
        for blk in self.blocks:
            a, b = blk(a, b)
        return torch.cat([a, b], dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        a, b = z[..., : self.d_a], z[..., self.d_a:]
        for blk in reversed(self.blocks):
            a, b = blk.inverse(a, b)
        return torch.cat([a, b], dim=-1)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class ReversibleLifting(nn.Module):
    """Invertible feature lifting + (non-invertible) masked temporal pooling.

    Input channels are ``[values (D), mask (D)]`` -- concatenating the mask is
    mandatory, otherwise the encoder cannot tell "lactate not measured" from
    "lactate measured as 0", and zero-padding would be read as a real value.
    """

    def __init__(self, n_markers: int, d_eps: int = 32, n_blocks: int = 3,
                 hidden: int = 64, static_dim: int = 0) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.d_in = 2 * n_markers
        self.lift = ZeroPadLifting(self.d_in, d_eps)
        self.stack = ReversibleStack(d_eps, n_blocks, hidden)
        self.static_dim = static_dim

        # Masked attention pooling over observed time points.
        self.pool_q = nn.Linear(d_eps, 1)
        self.head_in = d_eps + static_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.head_in),
            nn.Linear(self.head_in, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    # -- core ---------------------------------------------------------------
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) -> latent (..., d_eps). Invertible."""
        return self.stack(self.lift(x))

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        """Exact inverse of :meth:`encode_features`."""
        return self.lift.inverse(self.stack.inverse(z))

    def pool_time(self, z: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """Masked attention pooling. ``obs_mask``: (N, T) 1 = real time point."""
        scores = self.pool_q(z).squeeze(-1)                     # (N, T)
        neg = torch.full_like(scores, -1e4)
        scores = torch.where(obs_mask.bool(), scores, neg)
        w = torch.softmax(scores, dim=1)                        # (N, T)
        w = w * obs_mask
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (w.unsqueeze(-1) * z).sum(dim=1) / denom         # (N, d_eps)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        values / mask: (N, T, D). Returns logits (N,).
        """
        x = torch.cat([values, mask], dim=-1)
        z = self.encode_features(x)
        obs_mask = (mask.sum(dim=-1) > 0).float()               # (N, T)
        h = self.pool_time(z, obs_mask)
        if static is not None and self.static_dim > 0:
            h = torch.cat([h, static], dim=-1)
        return self.risk_head(h).squeeze(-1)

    # -- self-supervision ---------------------------------------------------
    def reconstruct(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct the *value* channels via the exact inverse."""
        x = torch.cat([values, mask], dim=-1)
        z = self.encode_features(x)
        x_hat = self.decode_features(z)
        return x_hat[..., : self.n_markers]
