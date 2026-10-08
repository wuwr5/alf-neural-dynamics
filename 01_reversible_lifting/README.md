# Component 1 — Reversible lifting / projection

**One sentence:** force the feature encoder to be *exactly invertible*, then use
that invertibility as a free self-supervised pretext task ("rebuild the labs"),
and only afterwards attach a Day-2 risk head.

## What it does

| Stage | Script | Uses labels? | Data used |
|---|---|---|---|
| 1 | `pretrain.py` | no | every row in `jm_long.csv` (907,046 rows) |
| 2 | `finetune.py` | yes | the Day-2 landmark risk set (n = 2,316, 783 events) |

### The block

Split the channel axis into `(a, b)`:

```
forward:   a' = a + g(b + f(a))
           b' = b + f(a)

inverse:   a  = a' - g(b')          # because b + f(a) = b'
           b  = b' - f(a)
```

The inverse is **closed form** — no fixed-point iteration, no extra capacity,
cost identical to the forward pass. Verified numerically in this repo:
`max |inverse(forward(x)) - x| = 0.00e+00` in float32.

Lifting is the IS-FNO choice: identity on the real channels + zero padding
(`ZeroPadLifting`), whose inverse is a slice. A learned linear lift is only
invertible if you constrain it; a zero pad is invertible by construction.

## Why it helps a small clinical cohort

1. **Information bottleneck as a prior.** An invertible encoder cannot silently
   discard its input. With 783 events, an unconstrained encoder will happily
   collapse the trajectory onto whatever correlates with the label *in this
   fold*; a constrained one must retain enough to reconstruct the labs.
2. **It unlocks the unlabelled data.** You have 907k longitudinal rows but only
   2,316 labelled landmark patients. Two-stage training is the standard way to
   spend that asymmetry.
3. **No decoder to tune.** Reconstruction is the architecture's own inverse, so
   there is no second network whose capacity becomes a hyperparameter.

## What it deliberately does *not* do

* It does **not** assume the clinical course is reversible. Death is an
  absorbing state and the ICU trajectory is dissipative. The reversible
  constraint applies only to the *feature lifting* at each observed time step.
* Temporal pooling afterwards is **not** invertible (masked attention pooling).
  Forcing invertibility across time would be the part that makes no clinical
  sense.

## Required control arms

Before claiming anything, run all three and report the table:

```bash
python 01_reversible_lifting/pretrain.py --epochs 60 --markers core
python 01_reversible_lifting/finetune.py --init outputs/rev_encoder_core.pt --epochs 80
python 01_reversible_lifting/finetune.py --from-scratch --epochs 80 --tag rev_scratch
python 01_reversible_lifting/finetune.py --init outputs/rev_encoder_core.pt \
       --freeze-encoder --epochs 80 --tag rev_frozen
```

* `--from-scratch` is the **control**. Without it you cannot attribute any gain
  to the pretraining.
* `--freeze-encoder` is a linear probe: if pretrained > frozen-probe but
  pretrained ≈ scratch, the gain was optimisation, not representation.

## Notes / gotchas

* Input channels are `[values (D), mask (D)]`. **Never** feed values alone:
  zero padding would then be indistinguishable from a measured zero, and in ICU
  data "not measured" is itself prognostic (informative visit process).
* The per-marker z-scorer is fitted on the **training patients only** and only
  on rows inside the landmark window (`common/pipeline.py`).
* Competitive risk: liver transplantation defaults to `censor`
  (`--transplant-as exclude` for the sensitivity analysis).

## Outputs

`outputs/<tag>_{metrics,pred,dca}.csv` with DeLong AUC + 95% CI, Brier,
calibration slope/intercept, ICI, and a decision-curve net-benefit sweep — the
same metric set as the existing eleven-arm comparison, so the arm is directly
comparable.
