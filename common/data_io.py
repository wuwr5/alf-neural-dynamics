"""Cohort loading and landmark splitting for the ALF-ICU longitudinal cohort.

Expected inputs (produced upstream by ``40_extract_clean.py``):

``jm_long.csv``
    Long format longitudinal labs.
    Columns: ``id, marker, time_d, valuenum, log_value``
    ``time_d`` is days since **ICU admission** (t0 = ICU entry).

``jm_surv.csv``
    One row per patient.
    Columns: ``id, event_time_d, event, Liver_transplantation, Vasopressin,
    rrt, Age, gender_male``
    ``event == 1`` = in-hospital death; ``event == 0`` = censored (discharge).
    ``event_time_d`` = time to death or to censoring, whichever came first.

Design notes that matter for reproducibility
--------------------------------------------
* **Landmarking.** At landmark ``L`` (default 2.0 d) we keep only patients who
  are still under follow-up at ``L``, i.e. ``event_time_d >= L``. The outcome is
  then ``y = event`` restricted to that risk set. This is a *landmark* binary
  outcome: it avoids the immortal-time bias you get from conditioning on a
  post-baseline window without a risk-set restriction.
* **No future leakage.** Only measurements with ``time_d <= L`` are ever handed
  to the models. Everything after ``L`` is dropped at the loader level.
* **Competing risk.** Liver transplantation (104 patients) is a competing
  event. We offer two standard handlings and you must pick one and report it:
  ``transplant_as="censor"`` (treat as censored at transplant; estimates the
  cause-specific hazard arm) or ``transplant_as="exclude"`` (drop them).
  The default is ``censor`` because excluding deletes informative follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Markers used in the main JM analysis upstream (bilirubin + INR + creatinine).
CORE_MARKERS: List[str] = ["Bilirubin", "INR", "Creatinine"]

# All 19 markers present in jm_long.csv.
ALL_MARKERS: List[str] = [
    "ALT", "AST", "Albumin", "BUN", "Bicarbonate", "Bilirubin", "Calcium",
    "Chloride", "Creatinine", "Glucose", "Hemoglobin", "INR", "Lactate",
    "PT", "PTT", "Platelet", "Potassium", "Sodium", "WBC",
]

STATIC_COLS: List[str] = ["Age", "gender_male", "Vasopressin", "rrt"]


@dataclass
class LandmarkCohort:
    """Container returned by :func:`make_landmark_cohort`."""

    ids: np.ndarray              # (N,) patient ids, in cohort order
    y: np.ndarray                # (N,) binary landmark outcome
    event_time: np.ndarray       # (N,) event/censoring time measured from t0
    static: np.ndarray           # (N, S) static covariates
    static_names: List[str]
    landmark: float
    n_events: int

    def __len__(self) -> int:
        return len(self.ids)


def resolve_data_path(path: str) -> str:
    """Resolve a data path relative to the repo, then to its parent.

    The published repo expects ``data/`` next to the code, but in a working
    checkout the cohort usually lives one level up next to sibling projects.
    Trying both keeps the scripts runnable without editing defaults.
    """
    import os

    if os.path.exists(path):
        return path
    parent = os.path.join("..", path)
    if os.path.exists(parent):
        return parent
    raise FileNotFoundError(
        f"Could not find {path!r} (also tried {parent!r}). "
        "Point --long/--surv at the CSVs, see data/README.md."
    )


def load_long(path: str, use_log: bool = True) -> pd.DataFrame:
    """Load the long-format lab table.

    Parameters
    ----------
    use_log:
        If True the model input is the ``log_value`` column (already produced
        upstream), otherwise the raw ``valuenum``. Labs are right-skewed, so
        the log scale is the default; the scaler in :mod:`common.tensorize`
        additionally z-scores per marker.
    """
    df = pd.read_csv(resolve_data_path(path))
    value_col = "log_value" if (use_log and "log_value" in df.columns) else "valuenum"
    out = df[["id", "marker", "time_d", value_col]].rename(columns={value_col: "value"})
    return out.dropna(subset=["value"]).reset_index(drop=True)


def load_surv(path: str) -> pd.DataFrame:
    """Load the per-patient time-to-event table."""
    return pd.read_csv(resolve_data_path(path))


def make_landmark_cohort(
    surv: pd.DataFrame,
    landmark: float = 2.0,
    transplant_as: str = "censor",
    static_cols: Sequence[str] = STATIC_COLS,
    min_followup: Optional[float] = None,
) -> LandmarkCohort:
    """Restrict to the landmark risk set and assemble static covariates.

    Parameters
    ----------
    landmark:
        Landmark time in days since ICU admission.
    transplant_as:
        ``"censor"`` (default) or ``"exclude"`` -- see module docstring.
    min_followup:
        Optional further restriction, e.g. require ``event_time_d >= L + h``
        when evaluating a fixed horizon ``h``.
    """
    if transplant_as not in {"censor", "exclude"}:
        raise ValueError("transplant_as must be 'censor' or 'exclude'")

    df = surv.copy()

    if transplant_as == "exclude":
        df = df[df["Liver_transplantation"] != 1].copy()
    else:
        # Censor at transplant: transplant is not the event of interest, so the
        # patient leaves the risk set without contributing a death.
        tx = df["Liver_transplantation"] == 1
        df.loc[tx, "event"] = 0

    # Landmark risk set: still under follow-up at L.
    keep = df["event_time_d"] >= landmark
    if min_followup is not None:
        keep &= df["event_time_d"] >= min_followup
    df = df[keep].copy()

    df = df.sort_values("id").reset_index(drop=True)

    cols = [c for c in static_cols if c in df.columns]
    static = df[cols].to_numpy(dtype=np.float32)
    # Median-impute any residual missing static value (fit on the whole cohort
    # here; move this inside the CV fold if you want strict fold-level hygiene).
    med = np.nanmedian(static, axis=0)
    idx = np.where(np.isnan(static))
    static[idx] = np.take(med, idx[1])

    return LandmarkCohort(
        ids=df["id"].to_numpy(),
        y=df["event"].to_numpy(dtype=np.int64),
        event_time=df["event_time_d"].to_numpy(dtype=np.float32),
        static=static,
        static_names=list(cols),
        landmark=float(landmark),
        n_events=int(df["event"].sum()),
    )


def cohort_summary(cohort: LandmarkCohort) -> str:
    """One-line human-readable description of the landmark risk set."""
    n = len(cohort)
    rate = cohort.y.mean() if n else float("nan")
    return (
        f"landmark={cohort.landmark}d  n={n}  events={cohort.n_events} "
        f"({rate:.1%})  EPV={cohort.n_events / 10:.0f} per 10 params"
    )
