"""Turn irregular long-format labs into padded tensors *without* leaking the future.

The representation is the standard "observation set" format used by
GRU-ODE-Bayes / mTAN / Latent ODE:

* ``values``  : (N, T, D) -- observed values, **0 where unobserved**
* ``mask``    : (N, T, D) -- 1 where observed, 0 elsewhere
* ``times``   : (N, T)    -- the shared per-patient time grid, in days
* ``dt``      : (N, T)    -- ``times[t] - times[t-1]``, ``dt[:, 0] = 0``
* ``lengths`` : (N,)      -- number of real time points per patient
* ``static``  : (N, S)    -- static covariates

Why the shared time grid is per-patient rather than a fixed global grid
----------------------------------------------------------------------
A lab panel draws many markers at the same timestamp, so the union of
measurement times per patient is much shorter than the number of rows. In this
cohort the union within the first 2 days is 11 time points at the median
(P90 = 24, max = 50). Padding to a patient-specific length keeps the sequence
short and avoids forcing every patient onto a grid that is mostly empty.

Never zero-fill without a mask
------------------------------
Zero in ``values`` means "padding", it does **not** mean "measured zero". The
mask is a model input, and ``dt`` is a model input too, because in ICU data the
*sampling intensity itself is prognostic* (informative visit process). Anything
that throws those two channels away is encoding "not measured" as a real value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrajectoryBatch:
    """Padded irregular-trajectory tensors."""

    values: np.ndarray    # (N, T, D) float32
    mask: np.ndarray      # (N, T, D) float32
    times: np.ndarray     # (N, T)    float32
    dt: np.ndarray        # (N, T)    float32
    lengths: np.ndarray   # (N,)      int64
    static: np.ndarray    # (N, S)    float32
    marker_names: List[str]

    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def n_markers(self) -> int:
        return len(self.marker_names)

    def to_tensors(self):
        """Return a dict of ``torch.Tensor`` copies (lazy torch import)."""
        import torch

        return {
            "values": torch.as_tensor(self.values),
            "mask": torch.as_tensor(self.mask),
            "times": torch.as_tensor(self.times),
            "dt": torch.as_tensor(self.dt),
            "lengths": torch.as_tensor(self.lengths),
            "static": torch.as_tensor(self.static),
        }


class MarkerScaler:
    """Per-marker z-scoring fitted on *observed* values of the training fold only.

    Fitting the scaler on the whole dataset is a mild, commonly-made leak. It is
    not usually fatal, but since this repo is about clean dynamic prediction we
    fit on the training split and apply to validation/test.
    """

    def __init__(self) -> None:
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, long_df: pd.DataFrame, markers: Sequence[str]) -> "MarkerScaler":
        for m in markers:
            v = long_df.loc[long_df["marker"] == m, "value"].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            mu = v.mean() if v.size else 0.0
            sd = v.std(ddof=0) if v.size else 1.0
            self.mean_[m] = float(mu)
            self.std_[m] = float(sd) if sd > 1e-8 else 1.0
        return self

    def transform(self, long_df: pd.DataFrame) -> pd.DataFrame:
        out = long_df.copy()
        mu = out["marker"].map(self.mean_).astype(float)
        sd = out["marker"].map(self.std_).astype(float)
        out["value"] = (out["value"].astype(float) - mu) / sd
        return out

    def fit_transform(self, long_df: pd.DataFrame, markers: Sequence[str]) -> pd.DataFrame:
        return self.fit(long_df, markers).transform(long_df)


def build_trajectories(
    long_df: pd.DataFrame,
    ids: Sequence,
    markers: Sequence[str],
    landmark: float,
    max_obs: int = 64,
    time_round: int = 4,
) -> TrajectoryBatch:
    """Build padded trajectory tensors restricted to ``time_d <= landmark``.

    Parameters
    ----------
    long_df:
        Columns ``id, marker, time_d, value`` (ideally already z-scored).
    ids:
        Patient ids in the desired cohort order (``LandmarkCohort.ids``).
    max_obs:
        Hard cap on the number of time points per patient. The cohort reaches
        50 within the first 2 days, so 64 is safe; longer windows need more.
        Time points are **not** subsampled -- if a patient exceeds the cap we
        keep the earliest ``max_obs`` and warn via the returned lengths.
    time_round:
        Decimals used to snap timestamps so that a panel drawn at "the same
        time" collapses to one grid point.
    """
    markers = list(markers)
    m_index = {m: i for i, m in enumerate(markers)}
    id_index = {pid: i for i, pid in enumerate(ids)}

    sub = long_df[long_df["marker"].isin(markers)]
    sub = sub[sub["time_d"] <= landmark]
    sub = sub[sub["id"].isin(id_index)]
    if len(sub) == 0:
        raise ValueError("No longitudinal rows left after landmark/marker filtering.")

    sub = sub.copy()
    sub["time_key"] = sub["time_d"].round(time_round)
    # A marker measured twice in the same snapped timestamp -> mean.
    sub = (
        sub.groupby(["id", "time_key", "marker"], as_index=False)["value"]
        .mean()
        .sort_values(["id", "time_key"])
    )

    N, D = len(ids), len(markers)
    n_pts = np.zeros(N, dtype=np.int64)

    # Pass 1: per-patient time grid and length.
    grids: List[np.ndarray] = []
    for pid in ids:
        t = sub.loc[sub["id"] == pid, "time_key"].unique()
        t = np.sort(t)[:max_obs]
        grids.append(t)
        n_pts[id_index[pid]] = len(t)

    T = max(int(n_pts.max()), 1)
    values = np.zeros((N, T, D), dtype=np.float32)
    mask = np.zeros((N, T, D), dtype=np.float32)
    times = np.zeros((N, T), dtype=np.float32)

    # Pass 2: scatter values onto the grid.
    grouped = {pid: g for pid, g in sub.groupby("id")}
    for pid in ids:
        i = id_index[pid]
        g = grouped.get(pid)
        if g is None or n_pts[i] == 0:
            continue
        grid = grids[i]
        pos = {t: k for k, t in enumerate(grid)}
        tk = g["time_key"].to_numpy()
        mk = g["marker"].to_numpy()
        vl = g["value"].to_numpy(dtype=np.float32)
        keep = np.array([pos.get(t, -1) for t in tk])
        ok = keep >= 0
        rows = keep[ok]
        cols = np.array([m_index[m] for m in mk[ok]], dtype=np.int64)
        values[i, rows, cols] = vl[ok]
        mask[i, rows, cols] = 1.0
        times[i, : n_pts[i]] = grid

    dt = np.zeros_like(times)
    dt[:, 1:] = np.diff(times, axis=1)
    dt = np.clip(dt, 0.0, None).astype(np.float32)

    return TrajectoryBatch(
        values=values, mask=mask, times=times, dt=dt,
        lengths=n_pts, static=None, marker_names=markers,
    )


def attach_static(batch: TrajectoryBatch, static: np.ndarray) -> TrajectoryBatch:
    """Attach the (N, S) static matrix, keeping the cohort row order."""
    batch.static = np.asarray(static, dtype=np.float32)
    return batch


def train_valid_split(ids: np.ndarray, y: np.ndarray, seed: int = 20261008,
                      valid_frac: float = 0.25) -> tuple:
    """Stratified patient-level split (a patient never crosses folds)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    tr, va = [], []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_va = int(round(len(cls_idx) * valid_frac))
        va.extend(cls_idx[:n_va])
        tr.extend(cls_idx[n_va:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return np.array(sorted(tr)), np.array(sorted(va))
