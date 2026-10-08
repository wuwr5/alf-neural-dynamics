"""Component 1, stage 2 -- Day-2 landmark risk prediction on top of the encoder.

Two modes, and you should report both:

* ``--init <ckpt>``        two-stage: encoder pretrained on 907k unlabelled rows.
* ``--from-scratch``       identical architecture, random init **control arm**.

Without the from-scratch control you cannot claim the reversible pretraining
did anything -- that is the single most common way this kind of paper gets
rejected.

Also supports ``--freeze-encoder`` (linear probe): freeze the encoder and train
only the risk head. If two-stage beats frozen-probe but not from-scratch, the
gain is optimisation, not representation.

Run:
    python 01_reversible_lifting/finetune.py --init outputs/rev_encoder_core.pt
    python 01_reversible_lifting/finetune.py --from-scratch
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
from model import ReversibleLifting


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: landmark risk fine-tuning")
    add_common_args(p)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--init", default=None, help="stage-1 checkpoint")
    p.add_argument("--from-scratch", action="store_true",
                   help="ignore --init and train from random init (control arm)")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="linear probe: train only the risk head")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--tag", default="rev_core")
    return p.parse_args()


def load_pretrained(model: torch.nn.Module, path: str) -> None:
    ck = torch.load(path, map_location="cpu")
    src = ck["state_dict"]
    own = model.state_dict()
    n = 0
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            n += 1
    missing = [k for k in own if k not in src]
    model.load_state_dict(own, strict=False)
    print(f"[finetune] loaded {n} tensors from {path}; "
          f"randomly initialised: {missing}")


def to_dev(d: dict, dev) -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        else:
            out[k] = torch.as_tensor(np.asarray(v)).to(dev)
    return out


@torch.no_grad()
def predict(model, d, dev) -> np.ndarray:
    model.eval()
    logits = []
    bs = 256
    n = d["values"].shape[0]
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        sub = {k: (v[sl] if v is not None and v.shape[0] == n else v)
               for k, v in d.items() if k != "y"}
        logits.append(model(sub["values"], sub["mask"], sub["static"]).cpu())
    return torch.sigmoid(torch.cat(logits)).numpy()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    co = build_cohort(args)
    print(f"[finetune] {co.summary}")
    print(f"[finetune] markers={co.markers}  T_max={co.batch.values.shape[1]}  "
          f"train={len(co.tr)}  valid={len(co.va)}")

    model = ReversibleLifting(
        n_markers=len(co.markers), d_eps=args.latent_dim, n_blocks=args.n_blocks,
        hidden=args.hidden, static_dim=len(co.static_names),
    ).to(dev)

    mode = "from-scratch"
    if args.init and not args.from_scratch:
        load_pretrained(model, args.init)
        mode = f"pretrained({os.path.basename(args.init)})"
    if args.freeze_encoder:
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith("risk_head")
        mode += " + frozen encoder"

    dtr = to_dev(co.subset(co.tr), dev)
    dva = to_dev(co.subset(co.va), dev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    ytr = dtr["y"]
    pos_w = float((ytr.numel() - ytr.sum()) / ytr.sum().clamp_min(1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))

    best_auc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(co.tr))
        for s in range(0, len(perm), args.batch_size):
            sel = perm[s:s + args.batch_size]
            idx = torch.as_tensor(sel).to(dev)
            sub = {k: (v[idx] if v is not None and v.shape[0] == len(co.tr) else v)
                   for k, v in dtr.items() if k != "y"}
            logit = model(sub["values"], sub["mask"], sub["static"])
            loss = lossf(logit, dtr["y"][idx].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        p_va = predict(model, dva, dev)
        m = evaluate_all(co.y[co.va], p_va)
        flag = ""
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
            flag = " *"
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or bad >= args.patience:
            print(f"[finetune] ep {ep:>3}  AUC {m['auc']:.3f}  "
                  f"Brier {m['brier']:.4f}  slope {m['calib_slope']:.2f}{flag}")
        if bad >= args.patience:
            print(f"[finetune] early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    p_va = predict(model, dva, dev)
    m = evaluate_all(co.y[co.va], p_va)
    nb = net_benefit(co.y[co.va], p_va)

    print("\n=== validation (Day-2 landmark) ===")
    print(f"  mode        : {mode}")
    print(f"  n           : {len(co.va)}   events: {int(co.y[co.va].sum())}")
    print(f"  AUC         : {m['auc']:.3f} (95% CI {m['ci_low']:.3f}-{m['ci_high']:.3f})")
    print(f"  Brier       : {m['brier']:.4f}")
    print(f"  calib slope : {m['calib_slope']:.2f}   intercept {m['calib_intercept']:+.2f}")
    print(f"  ICI         : {m['ici']:.4f}")

    pd.DataFrame([{**m, "mode": mode, "n_valid": len(co.va),
                   "landmark": args.landmark, "markers": ",".join(co.markers)}]
                 ).to_csv(os.path.join(args.out_dir, f"{args.tag}_metrics.csv"), index=False)
    pd.DataFrame({"id": co.ids[co.va], "y": co.y[co.va], "p": p_va}
                 ).to_csv(os.path.join(args.out_dir, f"{args.tag}_pred.csv"), index=False)
    pd.DataFrame({"threshold": nb["threshold"], "net_benefit": nb["net_benefit"]}
                 ).to_csv(os.path.join(args.out_dir, f"{args.tag}_dca.csv"), index=False)
    print(f"[finetune] saved -> {args.out_dir}/{args.tag}_{{metrics,pred,dca}}.csv")


if __name__ == "__main__":
    main()
