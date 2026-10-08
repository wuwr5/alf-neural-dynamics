# Component 2 — Latent-space structured dynamics

**One sentence:** the transferable half of IS-FNO's "exponential Fourier layer"
idea is *not* the Fourier part — it is "put a structured state-transition in the
latent space instead of letting an RNN improvise". On irregular, sparse clinical
trajectories the legitimate instantiations of that idea are mTAN, Latent ODE and
GRU-ODE, and that is what this directory implements.

## The three models

| Model | Reference | What evolves | Handles irregular sampling by |
|---|---|---|---|
| `mTAN` | Shukla & Marlin, 2021 | nothing — it *interpolates* | attention onto learned reference times |
| `latent_ode` | Rubanova et al., 2019 | a latent state `z`, integrated 0 → landmark | continuous ODE + decay-aware encoder |
| `gru_ode` | De Brouwer et al., 2019 | a hidden state `h`, continuous GRU flow | ODE between observations, jump at them |

### mTAN

Learns `n_ref` reference time points. For each reference point `r_j` and each
observed time `t_i`:

```
A = softmax_j( Q(r_j) . K(t_i, x_i) / sqrt(k) )      # over observed i
v_hat_j = sum_i A_ji * x_i        # interpolated labs
m_hat_j = sum_i A_ji * mask_i     # interpolated missingness
```

The output is a **fixed-length** vector no matter how irregular the input, and
the interpolated mask is a first-class feature — without it the model cannot
tell "reference point in a data void" from "reference point with a normal value".

### Latent ODE

```
encoder (backwards GRU with exp decay)  ->  q(z0) = N(mu, sigma)
neural ODE  dz/dt = f(z)                ->  integrate z0 to the landmark L
risk head(z_L, static)
```

We train it **discriminatively**: no decoder, no ELBO, no KL. The original
Latent ODE is a VAE; adding a KL without a decoder collapses the latent, and
adding a decoder spends capacity generating labs, which is not the clinical
question.

### GRU-ODE (-lite)

Between observations:

```
r     = sigmoid(W_r x + U_r h)
z     = sigmoid(W_z x - U_z h)          # negated U, as in the paper
h_hat = tanh(W_h x + U_h (r * h))
dh/dt = z * (h_hat - h)
```

driven by `x` = the last observed vector, held piecewise-constant. Missing
dimensions sit at 0, which in the z-scored space is the **cohort mean** — an
explicit mean imputation, not a fake zero. At each real observation a GRU cell
performs the discrete jump.

> **This is the "-lite" version.** The real GRU-ODE-Bayes replaces the jump with
> a closed-form Kalman/Bayesian update carrying a per-observation noise model.
> Ours is a cheap, fair baseline; use the authors' code if you need the full
> model.

## Why not a Fourier / spectral layer here

The spectral layer's inductive bias is that the input lives on a translation-
invariant, periodic domain. Neither axis of the `[patient x time x marker]`
tensor qualifies:

* **marker axis** — reordering the 19 labs changes the spectrum, so the "modes"
  are an artefact of a column ordering you chose;
* **time axis** — non-periodic, non-stationary, censored, and interrupted by
  interventions (RRT, transfusion, vasopressors).

There is also no "resolution" to be invariant to. Discretisation convergence —
FNO's headline property — has no counterpart here.

## Running

```bash
# one arm
python 02_latent_dynamics/train_landmark.py --model mtan --epochs 120
# head-to-head, identical protocol
python 02_latent_dynamics/train_landmark.py --model all --epochs 120
# with the adaptive solver (needs: pip install torchdiffeq)
python 02_latent_dynamics/train_landmark.py --model latent_ode --solver dopri5
```

`--model all` also fits **LOCF + logistic regression** as a floor. Keep it: any
temporal model that cannot beat LOCF+LR on 783 events is not earning its
parameters, and that is the first thing a reviewer will ask for.

Smoke run on this cohort (3 core markers, 6 epochs, untuned — do not quote):

```
          model    auc   brier   calib_slope
LOCF+LR (floor) 0.7536  0.1842        1.08
           mtan 0.7112  0.2249        1.45
        gru_ode 0.6895  0.2470       35.51
     latent_ode 0.6884  0.2547       91.86
```

Read that table honestly: at 783 events the heavy latent-dynamics arms start
*behind* a last-observation logistic regression. That is the expected direction
and is exactly why these go in as exploratory arms with the floor reported.

## Solver note

`odeint.py` ships a self-contained RK4 so nothing beyond `torch` is required.
`--solver dopri5` uses `torchdiffeq` if installed. Dynamics are **autonomous**:
`f(t, z)` accepts `t` but ignores it, because an explicit time dependence lets
the flow memorise the sampling schedule — and in ICU data the sampling
intensity is itself prognostic, so that is a shortcut, not a feature.
