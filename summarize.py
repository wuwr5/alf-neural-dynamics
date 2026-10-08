"""Merge every ``*_metrics.csv`` produced by a run into one comparison table.

    python summarize.py --out-dir outputs/full

Component 3 writes relative-L2 (lower is better) rather than AUC, so it is
reported in a separate block instead of being ranked against the clinical arms.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd


def arm_from_filename(name: str) -> str:
    stem = os.path.basename(name)
    for suffix in ("_metrics.csv", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/full")
    ap.add_argument("--tag", default="full_summary.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        raise SystemExit(f"no such directory: {args.out_dir}")

    clinical, spectral = [], []
    for fn in sorted(os.listdir(args.out_dir)):
        if not fn.endswith("_metrics.csv"):
            continue
        path = os.path.join(args.out_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"[summarize] skip {fn}: {exc}")
            continue
        if df.empty:
            continue
        if "auc" in df.columns:
            df["arm"] = df.get("model", arm_from_filename(fn))
            clinical.append(df)
        else:
            df["arm"] = arm_from_filename(fn)
            spectral.append(df)

    if not clinical and not spectral:
        raise SystemExit(f"no *_metrics.csv found in {args.out_dir}")

    if clinical:
        clin = pd.concat(clinical, ignore_index=True)
        cols = ["arm", "auc", "ci_low", "ci_high", "brier",
                "calib_slope", "calib_intercept", "ici"]
        cols = [c for c in cols if c in clin.columns]
        clin = clin[cols].sort_values("auc", ascending=False)
        print("\n=== clinical arms, Day-2 landmark (validation) ===")
        print(clin.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        clin = pd.DataFrame()

    if spectral:
        spec = pd.concat(spectral, ignore_index=True)
        print("\n=== component 3, FNO on waveforms (relative L2, lower is better) ===")
        print(spec.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        rollout = os.path.join(args.out_dir, "fno_rollout.csv")
        if os.path.exists(rollout):
            r = pd.read_csv(rollout)
            print("\nrollout error growth:")
            print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not clin.empty:
        out = os.path.join(args.out_dir, args.tag)
        clin.to_csv(out, index=False)
        print(f"\n[summarize] saved -> {out}")


if __name__ == "__main__":
    main()
