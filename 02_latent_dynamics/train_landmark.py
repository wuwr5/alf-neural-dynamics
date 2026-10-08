"""Component 2 -- train the latent-dynamics arms on the Day-2 landmark.

Runs one, several, or all three models under an identical protocol (same tensors,
same split, same seed, same metrics) and writes a single comparison table, then
adds a last-observation-carried-forward logistic regression as a floor.

Run:
    python 02_latent_dynamics/train_landmark.py --model mtan --epochs 120
    python 02_latent_dynamics/train_landmark.py --model all --epochs 120
    python 02_latent_dynamics/train_landmark.py --model latent_ode --solver dopri5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.metrics import evaluate_all, net_benefit
from common.pipeline import add_common_args, build_cohort
from models import MODEL_REGISTRY, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Latent-dynamics arms, Day-2 landmark")
    add_common_args(p)
    p.add_argument("--model", default="all",
                   help="mtan | latent_ode | gru_ode | all | comma list")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-ref", type=int, default=16, help="mTAN reference time points")
    p.add_argument("--solver", default="rk4", choices=["rk4", "euler", "dopri5"])
    p.add_argument("--n-steps", type=int, default=8, help="ODE substeps per interval")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the LOCF logistic-regression floor")
    return p.parse_args()


def ref_init_from_data(batch, landmark: float, n_ref: int) -> torch.Tensor:
    """Initialise mTAN reference times at quantiles of the observed times."""
    times = batch.times.reshape(-1)
    lens = batch.lengths
    keep = np.concatenate([t[: l] for t, l in zip(batch.times, lens)])
    keep = keep[(keep > 0) & (keep <= landmark)]
    if keep.size < n_ref:
        return torch.linspace(0.0, landmark, n_ref)
    q = np.quantile(keep, np.linspace(0, 1, n_ref))
    return torch.as_tensor(q, dtype=torch.float32)


def to_dev(d: dict, dev) -> dict:
    return {k: (None if v is None else torch.as_tensor(np.asarray(v)).to(dev))
            for k, v in d.items()}


@torch.no_grad()
def predict(model, d, dev) -> np.ndarray:
    model.eval()
    out, bs = [], 256
    n = d["values"].shape[0]
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        sub = {k: (v[sl] if torch.is_tensor(v) and v.shape[0] == n else v)
               for k, v in d.items() if k != "y"}
        out.append(torch.sigmoid(model(**sub)).cpu())
    return torch.cat(out).numpy()


def train_one(name, co, args, dev, ref_init) -> dict:
    model = build_model(
        name, n_markers=len(co.markers), static_dim=len(co.static_names),
        landmark=args.landmark, latent_dim=args.latent_dim, hidden=args.hidden,
        n_ref=args.n_ref, solver=args.solver, n_steps=args.n_steps,
        ref_init=ref_init,
    ).to(dev)

    dtr = to_dev(co.subset(co.tr), dev)
    dva = to_dev(co.subset(co.va), dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    ytr = dtr["y"]
    pos_w = float((ytr.numel() - ytr.sum()) / ytr.sum().clamp_min(1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))

    best_auc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(args.seed)
    n_tr = len(co.tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, args.batch_size):
            idx = torch.as_tensor(perm[s:s + args.batch_size]).to(dev)
            sub = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n_tr else v)
                   for k, v in dtr.items() if k != "y"}
            loss = lossf(model(**sub), dtr["y"][idx].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        m = evaluate_all(co.y[co.va], predict(model, dva, dev))
        if m["auc"] > best_auc:
            best_auc, bad = m["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 20 == 0 or bad >= args.patience:
            print(f"[{name:>13}] ep {ep:>3}  AUC {m['auc']:.3f}  Brier {m['brier']:.4f}")
        if bad >= args.patience:
            print(f"[{name:>13}] early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    p_va = predict(model, dva, dev)
    m = evaluate_all(co.y[co.va], p_va)
    nb = net_benefit(co.y[co.va], p_va)

    tag = f"latdyn_{name}"
    pd.DataFrame([{**m, "model": name, "n_valid": len(co.va), "landmark": args.landmark,
                   "markers": ",".join(co.markers)}]).to_csv(
        os.path.join(args.out_dir, f"{tag}_metrics.csv"), index=False)
    pd.DataFrame({"id": co.ids[co.va], "y": co.y[co.va], "p": p_va}).to_csv(
        os.path.join(args.out_dir, f"{tag}_pred.csv"), index=False)
    pd.DataFrame({"threshold": nb["threshold"], "net_benefit": nb["net_benefit"]}).to_csv(
        os.path.join(args.out_dir, f"{tag}_dca.csv"), index=False)
    return {**m, "model": name}


def locf_baseline(co, args) -> dict:
    """Last-observation-carried-forward + logistic regression (the floor).

    Any temporal model that cannot beat LOCF+LR on 783 events is not earning
    its parameters, and reviewers will ask for exactly this comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    v, m = co.batch.values.copy(), co.batch.mask.copy()
    N, T, D = v.shape
    locf = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        L = int(co.batch.lengths[i])
        if L == 0:
            continue
        for d in range(D):
            obs = np.where(m[i, :L, d] > 0)[0]
            locf[i, d] = v[i, obs[-1], d] if obs.size else 0.0
    X = np.column_stack([locf, co.batch.static if co.batch.static is not None
                         else np.zeros((N, 0))])

    Xtr, ytr = X[co.tr], co.y[co.tr]
    Xva, yva = X[co.va], co.y[co.va]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xva))[:, 1]
    out = evaluate_all(yva, p)
    out["model"] = "LOCF+LR (floor)"
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    co = build_cohort(args)
    print(f"[latdyn] {co.summary}")
    print(f"[latdyn] markers={co.markers}  T_max={co.batch.values.shape[1]}  "
          f"train={len(co.tr)} valid={len(co.va)}")

    ref_init = ref_init_from_data(co.batch, args.landmark, args.n_ref)
    if args.model == "all":
        names = list(MODEL_REGISTRY)
    else:
        names = [n.strip() for n in args.model.split(",")]

    rows = []
    for nm in names:
        rows.append(train_one(nm, co, args, dev, ref_init))
    if not args.no_baseline:
        rows.append(locf_baseline(co, args))

    df = pd.DataFrame(rows)[["model", "auc", "ci_low", "ci_high", "brier",
                             "calib_slope", "calib_intercept", "ici"]]
    df = df.sort_values("auc", ascending=False)
    print("\n=== Day-2 landmark, validation (n=%d, events=%d) ===" %
          (len(co.va), int(co.y[co.va].sum())))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(args.out_dir, "latent_dynamics_comparison.csv"), index=False)
    print(f"\n[latdyn] saved -> {args.out_dir}/latent_dynamics_comparison.csv")


if __name__ == "__main__":
    main()
