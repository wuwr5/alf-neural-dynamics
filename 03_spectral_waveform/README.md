# Component 3 — Spectral / FNO global convolution (waveforms)

**One sentence:** the Fourier layer is the one piece of IS-FNO we do *not*
recommend moving to the lab trajectory — but it is genuinely the right tool for
**dense high-frequency physiological signals**, and this directory is where it
belongs.

## What the layer does

```
(K v)(x)  =  F^{-1}{ R_k . F{v}_k }   for |k| <= k_max
          +  local 1x1 convolution
```

FFT → keep the lowest `modes` → mix channels with a complex weight per mode →
inverse FFT. Two consequences:

* the kernel is **global** (one weight per frequency, not per lag);
* the operator is **discretisation-invariant** — train at 64 phase points, run
  at 256. That is FNO's headline property.

## The demo task

Beat-structured arterial-pressure-like waveform, `(n_beats, n_phase)`. The FFT
runs over the **phase** axis, which is the axis that really is quasi-periodic
(beats repeat). Input = 8 beats, target = the next 8 beats; then the prediction
is fed back in for an autoregressive rollout.

```bash
python 03_spectral_waveform/train_surrogate.py --epochs 60 --rollout 4 --plot
```

Smoke run on this machine (400 train / 80 test, 40 epochs, CPU):

```
        FNO             : 0.2561
        persistence     : 0.4385   (repeat last beat)
        predict zero    : 1.0000

rollout:  8 beats ahead 0.2464
         16 beats ahead 0.4539
         24 beats ahead 0.6026
```

Two things to read off that:

1. The operator clearly beats persistence — it is predicting the beat-to-beat
   amplitude evolution, not just smoothing.
2. Error **accumulates** under rollout (0.25 → 0.60). That is exactly the
   failure mode IS-FNO targets with its reversible lifting and exponential
   spectral layers. If you want to reproduce their claim on medical signals,
   this script is the harness to do it in.

## A trap we hit, worth knowing

The first version of the generator used `ar_a = 0.75` for the beat-to-beat
amplitude. Correlation between beat `b` and beat `b+8` is then `0.75**8 = 0.10`
— the future is essentially independent of the past, and FNO scored 0.88
against a trivial 1.00 for predicting zero. It looked like a broken model; it
was a broken *task*. We ship `ar_a = 0.97`.

Same lesson applies clinically: before blaming the architecture, check that the
future is actually determined by what you observed.

## Where to point this in a real project

MIMIC-IV Waveform Database — arterial blood pressure, ECG, PLETH, ventilator
pressure/flow loops, all at 125 Hz. Replace `synthetic_waveform.py` with a
reader that returns a `(n_beats, n_phase)` matrix per record; nothing else in
the training script changes.

Also provided: `SpectralConv2d` / `FNO2d` for genuine 2D field problems where
FNO is uncontroversial — imaging inverse problems (MRI reconstruction,
photoacoustic/ultrasound CT), cardiac electrophysiology propagation, blood flow
and radiotherapy dose fields.

## What this is *not* for

Do not point it at `[patient x time x marker]` lab tensors. Both axes fail the
translation-invariance assumption: the marker axis is an arbitrary column
ordering (permute the labs, the spectrum changes), and the time axis is
non-periodic, non-stationary, censored, and interrupted by interventions. There
is also no "resolution" to be invariant to. Use components 1 and 2 instead.
