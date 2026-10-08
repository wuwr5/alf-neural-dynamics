"""Component 2 -- latent-space structured dynamics for irregular trajectories.

Three models, all of which consume the same observation-set tensors
``(values, mask, times, dt, lengths)`` and none of which require a regular grid:

1. **mTAN** (multi-time attention, Shukla & Marlin 2021)
   Learns a set of reference time points and interpolates the irregular series
   onto them by time-attention, producing a *fixed-length* representation.
   Simplest of the three, and on small cohorts it is usually the strongest.

2. **Latent ODE** (Rubanova et al. 2019)
   An ODE-RNN encoder reads the history **backwards**, with an exponential decay
   between observations, to produce ``q(z0)``; a neural ODE then integrates
   ``z0`` forward to the landmark time. We train it discriminatively
   (no decoder, no ELBO) -- see the note below.

3. **GRU-ODE (-lite)** (De Brouwer et al. 2019)
   Between observations the hidden state follows a continuous GRU flow
   ``dh/dt = z_gate * (h_hat - h)`` driven by the piecewise-constant last
   observation; at each real observation a discrete GRU jump injects the new
   measurement.

Honest deviations from the originals
------------------------------------
* **Latent ODE is discriminative.** The original is a VAE with a reconstruction
  decoder and a KL term. We drop both and train the encoder + ODE + risk head
  directly on the landmark label. Adding a KL without a decoder collapses the
  latent; adding a decoder spends capacity on generating labs, which is not the
  clinical question.
* **GRU-ODE-Bayes "-lite".** The original replaces the discrete jump with a
  closed-form *Bayesian* (Kalman) update that carries a per-observation noise
  model. We use a gated GRU jump instead. If you need the full model, use the
  authors' implementation -- ours exists to be a fair, cheap baseline.
* **Autonomous dynamics.** ``f(t, z)`` accepts ``t`` but ignores it. Making the
  flow explicitly time-dependent lets it memorise the sampling schedule, which
  in ICU data is itself a risk marker and therefore a leakage-ish shortcut.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from odeint import solve


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def step_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(N,T) float mask, 1 for real time points, 0 for padding."""
    ar = torch.arange(T, device=lengths.device).unsqueeze(0)
    return (ar < lengths.unsqueeze(1)).float()


class RiskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, p_drop: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# --------------------------------------------------------------------------
# 1. mTAN
# --------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """Learnable continuous-time embedding: t -> R^k via an MLP on the scalar."""

    def __init__(self, k: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, k)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) -> (..., k)
        return self.net(t.unsqueeze(-1).float())


class mTAN(nn.Module):
    """Multi-time attention: irregular series -> fixed-length reference grid.

    ``A`` = softmax over observed time points of ``Q(ref) . K(obs) / sqrt(k)``,
    applied to both the values and the missingness mask. Concatenating the
    interpolated **mask** is not optional: it is what tells the head that a
    reference point sits in a data void rather than at a normal value.
    """

    name = "mTAN"

    def __init__(self, n_markers: int, static_dim: int = 0, latent_dim: int = 32,
                 hidden: int = 64, n_ref: int = 16, k: int = 32,
                 learnable_ref: bool = True, ref_init: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_ref = n_ref
        self.time_emb = TimeEmbedding(k)
        self.q_proj = nn.Linear(k, k, bias=False)

        if ref_init is None:
            ref_init = torch.linspace(0.0, 1.0, n_ref)
        self.ref_times = nn.Parameter(ref_init.float().clone(), requires_grad=learnable_ref)

        self.val_emb = nn.Linear(n_markers, k)
        self.head = RiskHead(n_ref * 2 * n_markers + static_dim, hidden)

    def forward(self, values, mask, times, dt, lengths, static=None) -> torch.Tensor:
        N, T, D = values.shape
        obs = step_mask(lengths, T).to(values.dtype)            # (N,T)

        K = self.time_emb(times) + self.val_emb(values * mask)  # (N,T,k)
        Q = self.q_proj(self.time_emb(self.ref_times))          # (n_ref,k)

        scores = torch.einsum("rk,ntk->nrt", Q, K) / (K.shape[-1] ** 0.5)
        scores = scores.masked_fill(obs.unsqueeze(1) < 0.5, -1e4)
        A = torch.softmax(scores, dim=-1)                       # (N,n_ref,T)
        A = A * obs.unsqueeze(1)
        denom = A.sum(-1, keepdim=True).clamp_min(1e-6)
        A = A / denom

        v_imp = torch.einsum("nrt,ntd->nrd", A, values * mask)  # (N,n_ref,D)
        m_imp = torch.einsum("nrt,ntd->nrd", A, mask)           # (N,n_ref,D)

        h = torch.cat([v_imp, m_imp], dim=-1).reshape(N, -1)
        if static is not None and static.shape[-1] > 0:
            h = torch.cat([h, static], dim=-1)
        return self.head(h)


# --------------------------------------------------------------------------
# 2. Latent ODE
# --------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """Autonomous neural ODE right-hand side."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ODERNNEncoder(nn.Module):
    """GRU with exponential time decay, run backwards over the observations."""

    def __init__(self, n_markers: int, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(2 * n_markers, hidden)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(hidden, 2 * latent_dim)

    def forward(self, values, mask, lengths, dt) -> Tuple[torch.Tensor, torch.Tensor]:
        N, T, _ = values.shape
        h = values.new_zeros(N, self.gru.hidden_size)
        decay_rate = F.softplus(self.gamma) + 1e-6
        for t in range(T - 1, -1, -1):
            x = torch.cat([values[:, t], mask[:, t]], dim=-1)
            gap = dt[:, t].clamp_min(0.0).unsqueeze(-1)
            h_dec = h * torch.exp(-decay_rate * gap)
            h_new = self.gru(x, h_dec)
            valid = (lengths > t).float().unsqueeze(-1)
            h = h_new * valid + h * (1.0 - valid)
        mu, logvar = self.fc(h).chunk(2, dim=-1)
        return mu, logvar


class LatentODE(nn.Module):
    """ODE-RNN encoder -> latent ODE integrated 0 -> landmark -> risk head."""

    name = "LatentODE"

    def __init__(self, n_markers: int, static_dim: int = 0, latent_dim: int = 32,
                 hidden: int = 64, solver: str = "rk4", n_steps: int = 8,
                 landmark: float = 2.0) -> None:
        super().__init__()
        self.encoder = ODERNNEncoder(n_markers, hidden, latent_dim)
        self.ode = ODEFunc(latent_dim, hidden)
        self.solver, self.n_steps, self.landmark = solver, n_steps, landmark
        self.latent_dim = latent_dim
        self.head = RiskHead(latent_dim + static_dim, hidden)

    def forward(self, values, mask, times, dt, lengths, static=None) -> torch.Tensor:
        mu, logvar = self.encoder(values, mask, lengths, dt)
        if self.training:
            z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar).clamp_max(3.0)
        else:
            z0 = mu
        span = torch.full((z0.shape[0], 1), float(self.landmark),
                          dtype=z0.dtype, device=z0.device)
        zL = solve(self.ode, z0, span, method=self.solver, n_sub=self.n_steps)
        h = zL if static is None or static.shape[-1] == 0 else torch.cat([zL, static], -1)
        return self.head(h)


# --------------------------------------------------------------------------
# 3. GRU-ODE (-lite)
# --------------------------------------------------------------------------
class GRUODECell(nn.Module):
    """Continuous GRU flow: dh/dt = z_gate * (h_hat - h)."""

    def __init__(self, n_markers: int, hidden: int) -> None:
        super().__init__()
        self.w_r = nn.Linear(n_markers, hidden)
        self.u_r = nn.Linear(hidden, hidden, bias=False)
        self.w_z = nn.Linear(n_markers, hidden)
        self.u_z = nn.Linear(hidden, hidden, bias=False)
        self.w_h = nn.Linear(n_markers, hidden)
        self.u_h = nn.Linear(hidden, hidden, bias=False)

    def forward(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.w_r(x) + self.u_r(h))
        z = torch.sigmoid(self.w_z(x) - self.u_z(h))     # negated U, as in the paper
        h_hat = torch.tanh(self.w_h(x) + self.u_h(r * h))
        return z * (h_hat - h)


class GRUODEBayesLite(nn.Module):
    """Continuous GRU-ODE flow between observations + gated GRU jump at them.

    Missing dimensions are held at 0 between observations, which in the
    z-scored space used here means "the cohort mean" -- an explicit
    mean-imputation, not a spurious zero.
    """

    name = "GRU-ODE-lite"

    def __init__(self, n_markers: int, static_dim: int = 0, latent_dim: int = 32,
                 hidden: int = 64, solver: str = "rk4", n_steps: int = 4,
                 landmark: float = 2.0) -> None:
        super().__init__()
        self.n_markers, self.hidden = n_markers, hidden
        self.cell = GRUODECell(n_markers, hidden)
        self.jump = nn.GRUCell(2 * n_markers, hidden)
        self.solver, self.n_steps, self.landmark = solver, n_steps, landmark
        self.head = RiskHead(hidden + static_dim, hidden)

    def _flow(self, h: torch.Tensor, x: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        f = lambda t, hh: self.cell(t, hh, x)  # noqa: E731
        return solve(f, h, span, method=self.solver, n_sub=self.n_steps)

    def forward(self, values, mask, times, dt, lengths, static=None) -> torch.Tensor:
        N, T, D = values.shape
        h = values.new_zeros(N, self.hidden)
        x_last = values.new_zeros(N, D)

        for t in range(T):
            valid = (lengths > t).float().unsqueeze(-1)          # (N,1)
            # discrete jump: inject the measurement at time t
            x_jump = torch.cat([values[:, t] * mask[:, t], mask[:, t]], dim=-1)
            h = self.jump(x_jump, h) * valid + h * (1.0 - valid)
            # mean-impute the running observation vector
            x_last = torch.where(mask[:, t].bool(), values[:, t], x_last)
            x_last = x_last * valid + x_last * (1.0 - valid)
            # continuous flow to the next observed time
            if t + 1 < T:
                span = dt[:, t + 1].clamp_min(0.0).unsqueeze(-1)
                h = self._flow(h, x_last, span)

        # integrate from the last observation to the landmark
        last_t = times.gather(1, (lengths.clamp_min(1) - 1).unsqueeze(1)).clamp_min(0.0)
        span = (float(self.landmark) - last_t).clamp_min(0.0)
        h = self._flow(h, x_last, span)

        if static is not None and static.shape[-1] > 0:
            h = torch.cat([h, static], dim=-1)
        return self.head(h)


# --------------------------------------------------------------------------
MODEL_REGISTRY = {"mtan": mTAN, "latent_ode": LatentODE, "gru_ode": GRUODEBayesLite}


def build_model(name: str, n_markers: int, static_dim: int, landmark: float,
                latent_dim: int = 32, hidden: int = 64, n_ref: int = 16,
                solver: str = "rk4", n_steps: int = 8,
                ref_init: Optional[torch.Tensor] = None) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"model must be one of {list(MODEL_REGISTRY)}, got {name!r}")
    if key == "mtan":
        return mTAN(n_markers, static_dim, latent_dim, hidden, n_ref, ref_init=ref_init)
    if key == "latent_ode":
        return LatentODE(n_markers, static_dim, latent_dim, hidden, solver, n_steps, landmark)
    return GRUODEBayesLite(n_markers, static_dim, latent_dim, hidden, solver,
                           max(2, n_steps // 2), landmark)
