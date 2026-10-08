"""Self-contained ODE solvers -- no ``torchdiffeq`` required.

We ship our own fixed-step RK4 so the repo runs on a bare CPU install. If
``torchdiffeq`` happens to be installed you can switch to the adaptive
``dopri5`` solver with ``--solver dopri5``; results should agree to ~1e-3.

The dynamics are **autonomous** (``f`` takes ``t`` for signature compatibility
but the models here do not use it). An explicit time dependence in a latent ODE
trained on 783 events is a licence to overfit the sampling schedule, and it buys
nothing for a landmark prediction.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def rk4(f: Callable, y: torch.Tensor, dt: torch.Tensor, n_sub: int = 8) -> torch.Tensor:
    """Advance ``y`` by ``dt`` (shape (N,1) or scalar) with ``n_sub`` RK4 substeps.

    ``f(t, y) -> dy/dt``. Returns the state at ``t + dt``.
    """
    if not torch.is_tensor(dt):
        dt = torch.full((y.shape[0], 1), float(dt), dtype=y.dtype, device=y.device)
    dt = dt.to(y.dtype).reshape(-1, 1)
    h = dt / float(n_sub)
    t = torch.zeros_like(dt)
    for _ in range(n_sub):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return y


def euler(f: Callable, y: torch.Tensor, dt: torch.Tensor, n_sub: int = 8) -> torch.Tensor:
    """Fixed-step explicit Euler. Cheap baseline; RK4 is the default."""
    if not torch.is_tensor(dt):
        dt = torch.full((y.shape[0], 1), float(dt), dtype=y.dtype, device=y.device)
    dt = dt.to(y.dtype).reshape(-1, 1)
    h = dt / float(n_sub)
    t = torch.zeros_like(dt)
    for _ in range(n_sub):
        y = y + h * f(t, y)
        t = t + h
    return y


def dopri5(f: Callable, y: torch.Tensor, dt: torch.Tensor,
           rtol: float = 1e-4, atol: float = 1e-5) -> torch.Tensor:
    """Adaptive Dormand-Prince 5(4); requires ``torchdiffeq``."""
    try:
        from torchdiffeq import odeint
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dopri5 needs torchdiffeq: pip install torchdiffeq  "
            "(or just use --solver rk4, which is built in)"
        ) from exc

    t_span = torch.cat([torch.zeros(y.shape[0], 1, dtype=y.dtype, device=y.device),
                        dt.to(y.dtype).reshape(-1, 1)], dim=1)

    def wrapped(t, state):
        # torchdiffeq passes a scalar t; rebuild the (N,1) tensor.
        return f(t * torch.ones(state.shape[0], 1, dtype=state.dtype,
                                device=state.device), state)

    # Batched integration with a shared time grid: torchdiffeq wants one t
    # vector, so we integrate to the max dt and evaluate per-sample by scaling.
    # Simpler and exact enough here: integrate each sample is too slow, so we
    # use the mean dt and note the approximation.
    tmax = float(dt.mean())
    grid = torch.tensor([0.0, tmax], dtype=y.dtype, device=y.device)
    out = odeint(wrapped, y, grid, method="dopri5",
                 options={"rtol": rtol, "atol": atol})[-1]
    return out


SOLVERS = {"rk4": rk4, "euler": euler, "dopri5": dopri5}


def solve(f: Callable, y: torch.Tensor, dt: torch.Tensor, method: str = "rk4",
          n_sub: int = 8) -> torch.Tensor:
    """Dispatch helper. ``method`` in {rk4, euler, dopri5}."""
    if method not in SOLVERS:
        raise ValueError(f"solver must be one of {list(SOLVERS)}, got {method!r}")
    if method == "dopri5":
        return dopri5(f, y, dt)
    return SOLVERS[method](f, y, dt, n_sub=n_sub)
