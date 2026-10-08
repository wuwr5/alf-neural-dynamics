# alf-neural-dynamics

Three **component-level** building blocks extracted from
*An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE
Learning* (Rixin Yu, arXiv:2512.19439, *J. Comput. Phys.* 563:115081, 2026),
adapted so they can be used on **irregular clinical longitudinal data**
(a MIMIC-style acute-liver-failure ICU cohort, Day-2 landmark prediction of
in-hospital death).

> **Why components and not the whole architecture.**
> IS-FNO is a *time-advancement operator* for continuous fields on regular
> grids, benchmarked on Kuramoto–Sivashinsky / KdV / KP over 1000–4000-step
> rollouts. A clinical trajectory is sparse, irregular, non-periodic, censored,
> and terminates in an absorbing state. Four of its core inductive biases fail
> simultaneously, so the honest move is to keep the three ideas that survive
> and drop the rest.

---

## The three components

| # | Directory | What it is | What it is for | Use on your labs? |
|---|---|---|---|---|
| 1 | [`01_reversible_lifting/`](01_reversible_lifting/) | Reversible (RevNet-style) lifting with an exact closed-form inverse | Small-sample regularisation + label-free pretraining on 907k rows | **Yes** |
| 2 | [`02_latent_dynamics/`](02_latent_dynamics/) | mTAN / Latent ODE / GRU-ODE — structured latent state transition | The irregular-sampling version of "exponential spectral evolution" | **Yes** |
| 3 | [`03_spectral_waveform/`](03_spectral_waveform/) | FNO Fourier layer (FFT → mode truncation → complex mix → IFFT) | Global kernel for **dense quasi-periodic signals** | **No** — waveforms only |

### 1. Reversible lifting — *an encoder that cannot throw information away*

IS-FNO forces a near-reversible pairing between its lifting and projection maps.
The physical motivation (the inverse scattering transform is exactly reversible)
does not transfer. The **statistical** consequence does:

```
forward:   a' = a + g(b + f(a))
           b' = b + f(a)

inverse:   a  = a' - g(b')      # because b + f(a) = b'
           b  = b' - f(a)       # closed form, no fixed-point iteration
```

* With 783 events, a free-form encoder collapses the trajectory onto whatever
  correlates with the label *in this fold*. A constrained one must retain enough
  to rebuild the labs.
* Because the inverse is free, "reconstruct the labs" is a pretext task that
  needs **no labels** — so it can consume all 907,046 longitudinal rows, not
  just the 2,316 labelled landmark patients.
* Verified in this repo: `max |inverse(forward(x)) - x| = 0.00e+00` in float32.

**Not** reversible: temporal pooling. Death is absorbing; only the *feature*
lifting is constrained.

### 2. Latent dynamics — *the transferable half of the "exponential Fourier layer"*

IS-FNO's exponential Fourier layer is, in essence, "put a structured
state-transition in the latent space instead of letting a network improvise".
That idea is sound. The Fourier *parameterisation* of it is not, because neither
axis of a lab tensor is translation-invariant:

* **marker axis** — reorder the 19 labs and the spectrum changes; the "modes"
  are an artefact of a column ordering you picked;
* **time axis** — non-periodic, non-stationary, censored, interrupted by RRT,
  transfusion and vasopressors.

The legitimate instantiations for irregular sampling are **mTAN** (attention
onto learned reference times), **Latent ODE** (ODE-RNN encoder, neural ODE
integrated to the landmark) and **GRU-ODE** (continuous GRU flow + jump at each
observation). All three are implemented, plus a LOCF+logistic-regression floor.

### 3. Spectral / FNO — *right tool, wrong data*

FFT → keep `k_max` modes → complex channel mix → IFFT. Global kernel,
discretisation-invariant. Correct for the cardiac **phase axis** of a
125 Hz arterial waveform (beats repeat). Wrong for labs. Shipped with a
beat-structured synthetic generator and a rollout harness that shows the error
accumulation IS-FNO was built to fight.

---

## Quick start

```bash
pip install -r requirements.txt
cp config.example.yml config.yml      # optional; CLI flags override

# verify everything runs (tiny epoch counts)
python run_smoke.py
```

### Component 1

```bash
python 01_reversible_lifting/pretrain.py --epochs 60
python 01_reversible_lifting/finetune.py --init outputs/rev_encoder_core.pt --epochs 80
python 01_reversible_lifting/finetune.py --from-scratch --epochs 80 --tag rev_scratch
```

### Component 2

```bash
python 02_latent_dynamics/train_landmark.py --model all --epochs 120
```

### Component 3

```bash
python 03_spectral_waveform/train_surrogate.py --epochs 60 --rollout 4 --plot
```

### Full run (all three components, one seed)

```bash
bash run_full.sh            # everything -> outputs/full, log -> outputs/full/full_run.log
python summarize.py --out-dir outputs/full
```

`run_full.sh` runs component 1 in **three modes** (pretrained / from-scratch
control / frozen-encoder probe), component 2 with all three latent-dynamics arms
plus the LOCF floor, and component 3 with a 4-horizon rollout, then calls
`summarize.py` to merge everything into `outputs/full/full_summary.csv`.

It uses **one seed**. For a paper, repeat the whole script across seeds and
report the spread, not just the point estimate.

---

## Cohort and protocol

| Item | Value |
|---|---|
| Source | MIMIC-style ALF-ICU cohort, t0 = ICU admission |
| Longitudinal table | 907,046 rows, 19 markers, 2,501 patients |
| Full cohort | 2,508 patients, 962 deaths (38.4%) |
| **Day-2 landmark risk set** | **2,316 patients, 783 events (33.8%)** |
| Sampling density in [0, 2] d | median 11 time points (P90 24, max 50) |
| Main markers | Bilirubin, INR, Creatinine (lactate 57.2% missing, excluded) |

Leakage controls that are enforced in code, not just documented:

* only measurements with `time_d <= landmark` ever reach a model
  (`common/tensorize.py`);
* the per-marker z-scorer is fitted on **training patients only** and only on
  rows inside the landmark window (`common/pipeline.py`);
* missingness is a **model input** (`mask` and `dt` channels) — never zero-fill
  alone, because in ICU data sampling intensity is itself prognostic
  (informative visit process);
* liver transplantation (104 patients) is a competing event: `--transplant-as
  censor` (default) or `exclude`;
* a patient never crosses the train/validation split.

Metrics are the same set used by the existing eleven-arm comparison, so any arm
added here is directly comparable: DeLong AUC with 95% CI, Brier score,
calibration slope/intercept, ICI, and a decision-curve net-benefit sweep.

---

## Expected results, stated in advance

Do not be surprised by this ordering on 783 events:

```
LOCF + logistic regression   ~0.75
mTAN                         ~0.71   (after a handful of epochs; improves with tuning)
GRU-ODE / Latent ODE         ~0.69
```

Heavier parameterisation loses at this sample size, and the floor is reported on
purpose. These are **exploratory arms**. If a latent-dynamics arm beats the
joint model or the FT-Transformer, the result needs a pre-registered protocol
and an external cohort before it is worth claiming.

---

## Layout

```
alf-neural-dynamics/
├── README.md                       this file
├── run_smoke.py                    end-to-end verification
├── run_full.sh / summarize.py      full run (one seed) + merged results table
├── requirements.txt / config.example.yml / LICENSE / .gitignore
├── common/
│   ├── data_io.py                  cohort loading, landmarking, competing risk
│   ├── tensorize.py                irregular -> (values, mask, times, dt) tensors
│   ├── metrics.py                  DeLong AUC, Brier, calibration, DCA
│   └── pipeline.py                 one-call cohort assembly shared by 1 and 2
├── 01_reversible_lifting/          invertible lifting + two-stage training
├── 02_latent_dynamics/             mTAN / Latent ODE / GRU-ODE + solver
└── 03_spectral_waveform/           FNO layers + waveform surrogate harness
```

Each component directory is self-contained and has its own README, so any one of
them can be lifted into a separate repository.

## Deviations from the original papers (read before citing)

* **Latent ODE** is trained discriminatively — no decoder, no ELBO, no KL.
* **GRU-ODE** is the "-lite" variant: a gated GRU jump replaces the closed-form
  Bayesian/Kalman update of GRU-ODE-Bayes.
* **Component 3** currently runs on synthetic waveforms; swap in a MIMIC-IV
  waveform reader before reporting any number.
* ODE dynamics are **autonomous** (`f(t, z)` ignores `t`): explicit time
  dependence lets the flow memorise the sampling schedule, which is a shortcut
  in ICU data, not a feature.

## License

MIT. Patient-level row data are never committed (see `.gitignore`).
