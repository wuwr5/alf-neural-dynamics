"""Component 3 -- train the FNO time-advancement operator on beat-structured signals.

Task: given ``n_in`` beats, predict the next ``n_out`` beats.

Two evaluations, and the second is the one IS-FNO cares about:

* **direct**   -- relative L2 of the single-shot prediction.
* **rollout**  -- feed the prediction back in and advance again; report relative
  L2 at each horizon. This is where unconstrained latent operators fall apart
  (error accumulates) and where adding physical structure is supposed to pay.

Run:
    python 03_spectral_waveform/train_surrogate.py --epochs 60
    python 03_spectral_waveform/train_surrogate.py --epochs 60 --rollout 4 --plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno import FNO1d
from synthetic_waveform import generate_waveform, make_dataset, standardise


def parse_args():
    p = argparse.ArgumentParser(description="FNO time-advancement on waveforms")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--n-in", type=int, default=8, help="input beats (channels)")
    p.add_argument("--n-out", type=int, default=8, help="predicted beats (channels)")
    p.add_argument("--n-phase", type=int, default=64, help="samples per beat")
    p.add_argument("--modes", type=int, default=16, help="Fourier modes kept")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--rollout", type=int, default=3, help="autoregressive horizons")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.norm(pred - true, dim=(-2, -1))
            / torch.linalg.norm(true, dim=(-2, -1)).clamp_min(1e-8))


def build_rollout_set(args, seed: int = 999):
    """Waveforms long enough for ``args.rollout`` recursive steps."""
    need = args.n_in + args.n_out * (args.rollout + 1)
    rng = np.random.default_rng(seed)
    W = np.empty((args.n_test, need, args.n_phase), dtype=np.float32)
    for i in range(args.n_test):
        W[i] = generate_waveform(n_beats=need, n_phase=args.n_phase,
                                 hr=float(rng.uniform(60, 110)),
                                 f_resp=float(rng.uniform(0.15, 0.35)),
                                 noise=args.noise,
                                 seed=int(rng.integers(1 << 30)))
    return W


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    Xtr, Ytr = make_dataset(args.n_train, n_beats=args.n_in + args.n_out + 8,
                            n_phase=args.n_phase, n_in=args.n_in, n_out=args.n_out,
                            seed=args.seed, noise=args.noise)
    Xte, Yte = make_dataset(args.n_test, n_beats=args.n_in + args.n_out + 8,
                            n_phase=args.n_phase, n_in=args.n_in, n_out=args.n_out,
                            seed=args.seed + 1, noise=args.noise)
    Xtr, Ytr, mu, sd = standardise(Xtr, Ytr)
    Xte = (Xte - mu) / sd
    Yte = (Yte - mu) / sd

    Xtr_t = torch.as_tensor(Xtr).to(dev)
    Ytr_t = torch.as_tensor(Ytr).to(dev)
    Xte_t = torch.as_tensor(Xte).to(dev)
    Yte_t = torch.as_tensor(Yte).to(dev)

    model = FNO1d(args.n_in, args.n_out, width=args.width, modes=args.modes,
                  n_layers=args.n_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[fno] params={n_par}  modes={args.modes}  width={args.width}  "
          f"layers={args.n_layers}  in={args.n_in} beats  out={args.n_out} beats")

    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(args.n_train)
        tot, nb = 0.0, 0
        for s in range(0, args.n_train, args.batch_size):
            idx = torch.as_tensor(perm[s:s + args.batch_size]).to(dev)
            pred = model(Xtr_t[idx])
            loss = rel_l2(pred, Ytr_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                tr = rel_l2(model(Xtr_t), Ytr_t).mean().item()
                te = rel_l2(model(Xte_t), Yte_t).mean().item()
            print(f"[fno] ep {ep:>3}  train rel-L2 {tot/max(nb,1):.4f} | "
                  f"train {tr:.4f}  test {te:.4f}")

    model.eval()
    with torch.no_grad():
        direct = rel_l2(model(Xte_t), Yte_t).mean().item()

    # Persistence baseline: repeat the last observed beat for every future beat.
    # Any operator that loses to this is not predicting, it is smoothing.
    persist = Xte_t[:, -1:].expand(-1, args.n_out, -1)
    persist_err = rel_l2(persist, Yte_t).mean().item()
    zero_err = rel_l2(torch.zeros_like(Yte_t), Yte_t).mean().item()

    print(f"\n[fno] direct {args.n_in}->{args.n_out} beats, test relative L2:")
    print(f"        FNO             : {direct:.4f}")
    print(f"        persistence     : {persist_err:.4f}   (repeat last beat)")
    print(f"        predict zero    : {zero_err:.4f}")

    # ---- autoregressive rollout ------------------------------------------
    W = build_rollout_set(args)
    W_t = torch.as_tensor((W - mu) / sd).to(dev)
    errs = []
    with torch.no_grad():
        cur = W_t[:, : args.n_in]
        for h in range(1, args.rollout + 1):
            nxt = model(cur)
            lo = args.n_in + (h - 1) * args.n_out
            hi = lo + args.n_out
            e = rel_l2(nxt, W_t[:, lo:hi]).mean().item()
            errs.append(e)
            cur = nxt                      # feed prediction back in
    print("[fno] rollout error growth (relative L2):")
    for h, e in enumerate(errs, start=1):
        print(f"        horizon {h} ({h*args.n_out} beats ahead): {e:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        with torch.no_grad():
            p = model(Xte_t[:1])[0, -1].cpu().numpy()
        t = Yte[0, -1]
        ax[0].plot(t, label="truth")
        ax[0].plot(p, label="FNO")
        ax[0].set_title("last predicted beat (direct)")
        ax[0].legend()
        ax[1].plot(range(1, len(errs) + 1), errs, marker="o")
        ax[1].set_xlabel("rollout horizon")
        ax[1].set_ylabel("relative L2")
        ax[1].set_title("error accumulation")
        fig.tight_layout()
        out = os.path.join(args.out_dir, "fno_waveform.png")
        fig.savefig(out, dpi=130)
        print(f"[fno] plot -> {out}")

    import pandas as pd
    pd.DataFrame({"horizon": range(1, len(errs) + 1), "rel_l2": errs}).to_csv(
        os.path.join(args.out_dir, "fno_rollout.csv"), index=False)
    pd.DataFrame([{"direct_rel_l2": direct, "n_params": n_par, "modes": args.modes,
                   "width": args.width, "n_layers": args.n_layers}]).to_csv(
        os.path.join(args.out_dir, "fno_metrics.csv"), index=False)
    print(f"[fno] saved -> {args.out_dir}/fno_{{metrics,rollout}}.csv")


if __name__ == "__main__":
    main()
