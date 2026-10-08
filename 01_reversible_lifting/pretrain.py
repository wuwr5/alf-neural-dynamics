"""Component 1, stage 1 -- self-supervised pretraining of the reversible encoder.

Objective: masked reconstruction of observed lab values through the *exact
inverse* of the lifting stack.

    x = [values, mask]  --lift-->  z  --inverse-->  x_hat
    loss = MSE over entries that were actually measured

Why this works as pretraining here
----------------------------------
* It needs no labels, so it can consume the whole 907,046-row longitudinal
  table, not just the 2,316 patients in the Day-2 landmark risk set.
* The task is "rebuild the labs", which forces the latent code to retain
  cross-marker structure (e.g. the bilirubin/INR/creatinine coupling) instead
  of collapsing onto whatever predicts death in this particular fold.
* Because the inverse is closed-form, the reconstruction costs one extra
  forward pass -- no decoder network to train, no separate capacity to tune.

Run:
    python 01_reversible_lifting/pretrain.py --epochs 60
    python 01_reversible_lifting/pretrain.py --markers all --pretrain-window 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_io import ALL_MARKERS, CORE_MARKERS, load_long
from common.tensorize import MarkerScaler, build_trajectories
from model import ReversibleLifting


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: reversible-lifting pretraining")
    p.add_argument("--long", default="data/jm_long.csv")
    p.add_argument("--markers", default="core")
    p.add_argument("--pretrain-window", type=float, default=2.0,
                   help="use observations up to this day for pretraining")
    p.add_argument("--max-obs", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--mask-ratio", type=float, default=0.3,
                   help="fraction of observed entries additionally hidden")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    markers = ALL_MARKERS if args.markers in {"all", "all_19"} else (
        CORE_MARKERS if args.markers == "core"
        else [m.strip() for m in args.markers.split(",")]
    )

    long_df = load_long(args.long, use_log=True)
    win = long_df[long_df["time_d"] <= args.pretrain_window]
    win = win[win["marker"].isin(markers)]
    ids = np.sort(win["id"].unique())
    print(f"[pretrain] {len(ids)} patients, {len(win)} rows, {len(markers)} markers, "
          f"window <= {args.pretrain_window}d")

    scaler = MarkerScaler().fit(win, markers)
    win = scaler.transform(win)
    batch = build_trajectories(win, ids, markers, args.pretrain_window,
                               max_obs=args.max_obs)

    model = ReversibleLifting(
        n_markers=len(markers), d_eps=args.latent_dim,
        n_blocks=args.n_blocks, hidden=args.hidden, static_dim=0,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    values = torch.as_tensor(batch.values).to(dev)
    mask = torch.as_tensor(batch.mask).to(dev)
    N = values.shape[0]

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        for s in range(0, N, args.batch_size):
            idx = torch.as_tensor(perm[s:s + args.batch_size]).to(dev)
            v = values[idx]
            m = mask[idx]
            obs = m.clone()

            # Hide a random subset of the observed entries from the encoder.
            drop = (torch.rand_like(obs) < args.mask_ratio) & obs.bool()
            m_in = torch.where(drop, torch.zeros_like(m), m)

            x = torch.cat([v * m_in, m_in], dim=-1)
            z = model.encode_features(x)
            x_hat = model.lift.inverse(model.stack.inverse(z))
            v_hat = x_hat[..., : len(markers)]

            loss = F.mse_loss(v_hat * obs, v * obs, reduction="sum") / obs.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item())
            nb += 1

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[pretrain] epoch {ep:>3}/{args.epochs}  recon-MSE {tot / max(nb,1):.4f}"
                  f"  ({time.time()-t0:.0f}s)")

    ckpt = os.path.join(args.out_dir, f"rev_encoder_{'all' if len(markers) > 3 else 'core'}.pt")
    torch.save(
        {"state_dict": model.state_dict(),
         "markers": markers,
         "d_eps": args.latent_dim,
         "n_blocks": args.n_blocks,
         "hidden": args.hidden,
         "scaler": {"mean": scaler.mean_, "std": scaler.std_},
         "window": args.pretrain_window},
        ckpt,
    )
    print(f"[pretrain] saved -> {ckpt}")

    # Sanity: how invertible is it numerically? Should be ~1e-6 on CPU/float32.
    model.eval()
    with torch.no_grad():
        x = torch.cat([values[:16] * mask[:16], mask[:16]], dim=-1)
        err = (model.decode_features(model.encode_features(x)) - x).abs().max().item()
    print(f"[pretrain] max |inverse(forward(x)) - x| = {err:.2e}  (numerical check)")


if __name__ == "__main__":
    main()
