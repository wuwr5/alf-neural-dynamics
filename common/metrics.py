"""Evaluation metrics matching the main ALF-ICU paper pipeline.

All functions are numpy-only and deterministic. Reported:

* ``auc_delong``  -- AUC with the DeLong (Sun & Xu) covariance-based SE and a
  normal-approximation 95% CI.
* ``brier_score`` -- mean squared error of the probability.
* ``calibration`` -- logistic recalibration slope/intercept plus a
  10-quantile-binned Integrated Calibration Index (ICI).
* ``net_benefit`` -- decision-curve analysis, threshold sweep.

Why these and not just AUC: a dynamic model that discriminates well can still
be badly calibrated, and a well-calibrated model can still have no clinical
utility. The upstream eleven-arm comparison reports all three, so anything added
as a twelfth arm has to be scored the same way or the comparison is meaningless.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def auc_delong(y_true: Sequence, y_score: Sequence) -> Dict[str, float]:
    """AUC, DeLong SE and a 95% CI.

    Implements the fast Sun & Xu formulation using the empirical placement
    values V10 (positives) and V01 (negatives); ties contribute 0.5.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    keep = np.isfinite(p)
    y, p = y[keep], p[keep]

    x = p[y == 1]   # positives
    z = p[y == 0]   # negatives
    m, k = len(x), len(z)
    if m == 0 or k == 0:
        return {"auc": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}

    zs, xs = np.sort(z), np.sort(x)
    zl = np.searchsorted(zs, x, side="left")
    zr = np.searchsorted(zs, x, side="right")
    V10 = (zl + 0.5 * (zr - zl)) / k                       # P(X > Y) + 0.5 P(=)

    xl = np.searchsorted(xs, z, side="left")
    xr = np.searchsorted(xs, z, side="right")
    V01 = ((m - xr) + 0.5 * (xr - xl)) / m                 # P(X > Y) + 0.5 P(=)

    auc = float(V10.mean())
    s10 = np.var(V10, ddof=1) if m > 1 else 0.0
    s01 = np.var(V01, ddof=1) if k > 1 else 0.0
    se = float(np.sqrt(max(s10 / m + s01 / k, 0.0)))
    return {"auc": auc, "se": se,
            "ci_low": float(auc - 1.96 * se), "ci_high": float(auc + 1.96 * se)}


def brier_score(y_true: Sequence, y_score: Sequence) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def calibration(y_true: Sequence, y_score: Sequence, n_bins: int = 10) -> Dict[str, float]:
    """Logistic-recalibration slope/intercept and a binned ICI.

    Slope < 1 means the model is over-fit (predictions too extreme);
    the intercept captures systematic over/under-prediction.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    lp = _logit(p)

    # Closed-form logistic fit of y on logit(p) via a couple of Newton steps
    # (avoids a sklearn dependency just for a 2-parameter fit).
    a, b = 0.0, 1.0
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.array([a, b], dtype=float)
    for _ in range(50):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-8, None)
        grad = X.T @ (y - mu)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break

    # Binned ICI.
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    ici, tot = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p <= hi) if hi == edges[-1] else (p >= lo) & (p < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        ici += n * abs(p[sel].mean() - y[sel].mean())
        tot += n

    return {
        "calib_intercept": float(beta[0]),
        "calib_slope": float(beta[1]),
        "ici": float(ici / tot) if tot else float("nan"),
    }


def net_benefit(y_true: Sequence, y_score: Sequence,
                thresholds: Sequence | None = None) -> Dict[str, np.ndarray]:
    """Decision-curve analysis.

    ``NB(t)`` = sensitivity-weighted benefit minus the harm of false positives,
    scaled by the odds at threshold ``t``. Returns the threshold grid and the
    net benefit for both the model and "treat none" (which is 0 by definition).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = np.asarray(thresholds, dtype=float)

    n = len(y)
    nb = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = (p >= t).astype(float)
        tp = float(np.sum(pred * y))
        fp = float(np.sum(pred * (1 - y)))
        nb[i] = tp / n - (fp / n) * (t / (1.0 - t))
    return {"threshold": thresholds, "net_benefit": nb}


def evaluate_all(y_true: Sequence, y_score: Sequence) -> Dict[str, float]:
    """Convenience: discrimination + calibration in one dict."""
    out: Dict[str, float] = {}
    out.update(auc_delong(y_true, y_score))
    out["brier"] = brier_score(y_true, y_score)
    out.update(calibration(y_true, y_score))
    return out
