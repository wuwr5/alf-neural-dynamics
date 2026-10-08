"""Reference baselines shared by component 2 and the comparison script.

Kept in ``common/`` so that the floor inside ``train_landmark.py`` and the one
used by ``compare_arms.py`` are literally the same code -- if they drifted apart
the paired DeLong test would compare two different baselines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def locf_matrix(values: np.ndarray, mask: np.ndarray,
                lengths: np.ndarray) -> np.ndarray:
    """Last observation carried forward -> one vector per patient.

    Patients with no observation at all stay at 0, which in the z-scored space
    used throughout this repo means "the cohort mean".
    """
    N, _, D = values.shape
    out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        L = int(lengths[i])
        if L == 0:
            continue
        for d in range(D):
            obs = np.where(mask[i, :L, d] > 0)[0]
            out[i, d] = values[i, obs[-1], d] if obs.size else 0.0
    return out


def locf_logistic(cohort, seed: int = 20261008) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LOCF + L2 logistic regression. Returns ``(ids, y, p)`` on the valid split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.column_stack([
        locf_matrix(cohort.batch.values, cohort.batch.mask, cohort.batch.lengths),
        cohort.batch.static if cohort.batch.static is not None
        else np.zeros((len(cohort.ids), 0)),
    ])
    Xtr, ytr = X[cohort.tr], cohort.y[cohort.tr]
    Xva, yva = X[cohort.va], cohort.y[cohort.va]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xva))[:, 1].astype(np.float64)
    return cohort.ids[cohort.va], yva, p
