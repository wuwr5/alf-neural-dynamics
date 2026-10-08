"""One-call cohort assembly shared by components 1 and 2.

Keeping this in one place is deliberate: if component 1 and component 2 saw
differently-built tensors, their head-to-head AUCs would be incomparable, which
is the whole point of a multi-arm comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .data_io import (
    ALL_MARKERS,
    CORE_MARKERS,
    LandmarkCohort,
    cohort_summary,
    load_long,
    load_surv,
    make_landmark_cohort,
)
from .tensorize import (
    MarkerScaler,
    TrajectoryBatch,
    attach_static,
    build_trajectories,
    train_valid_split,
)


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--long", default="data/jm_long.csv", help="long-format lab CSV")
    p.add_argument("--surv", default="data/jm_surv.csv", help="per-patient survival CSV")
    p.add_argument("--landmark", type=float, default=2.0, help="landmark time in days")
    p.add_argument("--markers", default="core",
                   help="'core' (Bilirubin/INR/Creatinine), 'all' (19) or a comma list")
    p.add_argument("--max-obs", type=int, default=64, help="cap on time points per patient")
    p.add_argument("--transplant-as", default="censor", choices=["censor", "exclude"])
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--valid-frac", type=float, default=0.25)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    return p


def resolve_markers(spec: str) -> List[str]:
    if spec == "core":
        return list(CORE_MARKERS)
    if spec in {"all", "all_19"}:
        return list(ALL_MARKERS)
    return [m.strip() for m in spec.split(",") if m.strip()]


@dataclass
class Cohort:
    """Everything a training script needs."""

    ids: np.ndarray
    y: np.ndarray
    static: np.ndarray
    static_names: List[str]
    markers: List[str]
    batch: TrajectoryBatch          # full cohort, padded
    tr: np.ndarray                  # train row indices
    va: np.ndarray                  # validation row indices
    landmark: float
    summary: str

    def subset(self, idx: np.ndarray) -> dict:
        """Slice the padded tensors to a set of cohort rows."""
        return {
            "values": self.batch.values[idx],
            "mask": self.batch.mask[idx],
            "times": self.batch.times[idx],
            "dt": self.batch.dt[idx],
            "lengths": self.batch.lengths[idx],
            "static": self.batch.static[idx] if self.batch.static is not None else None,
            "y": self.y[idx],
        }


def build_cohort(args) -> Cohort:
    markers = resolve_markers(args.markers)
    long_df = load_long(args.long, use_log=True)
    surv = load_surv(args.surv)

    cohort = make_landmark_cohort(
        surv, landmark=args.landmark, transplant_as=args.transplant_as
    )
    ids = cohort.ids

    # Scaler is fitted on the TRAIN patients only, and only on rows inside the
    # landmark window (later rows must never influence even the normalisation).
    tr_all, va_all = train_valid_split(ids, cohort.y, seed=args.seed,
                                       valid_frac=args.valid_frac)
    tr_ids = set(ids[tr_all].tolist())

    window = long_df[long_df["time_d"] <= args.landmark]
    scaler = MarkerScaler().fit(window[window["id"].isin(tr_ids)], markers)
    long_df = scaler.transform(long_df)

    batch = build_trajectories(long_df, ids, markers, args.landmark, max_obs=args.max_obs)
    attach_static(batch, cohort.static)

    return Cohort(
        ids=ids,
        y=cohort.y,
        static=cohort.static,
        static_names=cohort.static_names,
        markers=markers,
        batch=batch,
        tr=tr_all,
        va=va_all,
        landmark=args.landmark,
        summary=cohort_summary(cohort),
    )
