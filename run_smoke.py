"""One-command verification of all three components.

Runs everything with deliberately tiny epoch counts -- the point is that the
pipeline executes, not that the numbers are good.

    python run_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("component 1 / pretrain (reversible lifting)",
     ["01_reversible_lifting/pretrain.py", "--epochs", "3", "--batch-size", "256"],
     "outputs/rev_encoder_core.pt"),
    ("component 1 / finetune (Day-2 landmark)",
     ["01_reversible_lifting/finetune.py", "--init", "outputs/rev_encoder_core.pt",
      "--epochs", "12", "--patience", "6"],
     "outputs/rev_core_metrics.csv"),
    ("component 2 / latent dynamics (all three arms)",
     ["02_latent_dynamics/train_landmark.py", "--model", "all",
      "--epochs", "8", "--patience", "4"],
     "outputs/latent_dynamics_comparison.csv"),
    ("component 3 / FNO on beat-structured waveforms",
     ["03_spectral_waveform/train_surrogate.py", "--epochs", "20",
      "--n-train", "300", "--n-test", "60", "--rollout", "3"],
     "outputs/fno_metrics.csv"),
]


def main() -> int:
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    failures = []

    for label, cmd, expect in STEPS:
        print("\n" + "=" * 72)
        print(f"  {label}")
        print("=" * 72)
        r = subprocess.run([PY] + cmd, cwd=ROOT, env=env)
        produced = os.path.exists(os.path.join(ROOT, expect.replace("/", os.sep)))
        ok = r.returncode == 0 and produced
        print(f"--> {'PASS' if ok else 'FAIL'}  (exit {r.returncode}, "
              f"artifact {expect}: {'yes' if produced else 'no'})")
        if not ok:
            failures.append(label)

    print("\n" + "=" * 72)
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All components ran. Artifacts in outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
