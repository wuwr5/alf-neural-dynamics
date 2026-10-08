"""Paired DeLong comparison of two arms on the same validation patients.

Comparing two AUCs by eye-balling their confidence intervals is wrong when both
were computed on the same people -- the estimates are correlated. This does the
proper covariance-based paired test.

    # a saved arm against the LOCF+LR floor
    python compare_arms.py --a outputs/full/latdyn_mtan_pred.csv --vs-locf
    # any two saved prediction files
    python compare_arms.py --a outputs/full/latdyn_mtan_pred.csv \
                           --b outputs/full/rev_pretrained_pred.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.baselines import locf_logistic
from common.metrics import auc_delong_paired, evaluate_all
from common.pipeline import build_cohort


def load_pred(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"id", "y", "p"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}")
    return df[["id", "y", "p"]]


def parse_args():
    p = argparse.ArgumentParser(description="paired DeLong test between two arms")
    p.add_argument("--a", required=True, help="prediction CSV (id, y, p)")
    p.add_argument("--b", default=None, help="second prediction CSV")
    p.add_argument("--vs-locf", action="store_true",
                   help="compare against the LOCF+LR floor, rebuilt on the same split")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--valid-frac", type=float, default=0.25)
    p.add_argument("--long", default="data/jm_long.csv")
    p.add_argument("--surv", default="data/jm_surv.csv")
    p.add_argument("--landmark", type=float, default=2.0)
    p.add_argument("--markers", default="core")
    p.add_argument("--max-obs", type=int, default=64)
    p.add_argument("--transplant-as", default="censor")
    p.add_argument("--out-dir", default="outputs/full")
    p.add_argument("--csv", default=None, help="optional path to append the result")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = load_pred(args.a)

    if args.vs_locf or args.b is None:
        co = build_cohort(args)
        ids, y, p = locf_logistic(co, seed=args.seed)
        b = pd.DataFrame({"id": ids, "y": y, "p": p})
        label_b = args.label_b or "LOCF+LR (floor)"
    else:
        b = load_pred(args.b)
        label_b = args.label_b or os.path.basename(args.b).replace("_pred.csv", "")

    label_a = args.label_a or os.path.basename(args.a).replace("_pred.csv", "")

    # Align on patient id; the arms share the split so this is a safety check.
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(m) == 0:
        raise SystemExit("no overlapping ids -- the two arms used different splits")
    if len(m) != len(a):
        print(f"[compare] warning: {len(a) - len(m)} ids did not match; using {len(m)}")

    res = auc_delong_paired(m["y_a"].to_numpy(), m["p_a"].to_numpy(), m["p_b"].to_numpy())

    print(f"\n=== paired DeLong: {label_a} vs {label_b} ===")
    print(f"  n = {len(m)}   events = {int(m['y_a'].sum())}")
    print(f"  AUC {label_a:<22s} : {res['auc_a']:.4f}")
    print(f"  AUC {label_b:<22s} : {res['auc_b']:.4f}")
    print(f"  difference          : {res['diff']:+.4f}  (SE {res['se_diff']:.4f})")
    print(f"  z = {res['z']:+.3f}   p = {res['p_value']:.4f}")
    verdict = ("significant" if res["p_value"] < 0.05 else
               "NOT significant at 0.05")
    print(f"  -> {verdict}")

    if args.csv:
        row = {"arm_a": label_a, "arm_b": label_b, **res, "n": len(m)}
        path = args.csv
        if os.path.exists(path):
            pd.concat([pd.read_csv(path), pd.DataFrame([row])]).to_csv(path, index=False)
        else:
            pd.DataFrame([row]).to_csv(path, index=False)
        print(f"[compare] appended -> {path}")


if __name__ == "__main__":
    main()
