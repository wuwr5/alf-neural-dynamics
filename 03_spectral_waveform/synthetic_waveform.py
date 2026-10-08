"""Synthetic beat-structured physiological waveform generator.

This exists for one reason: the FNO demo needs a signal whose Fourier
assumption is actually true. A synthetic arterial-pressure-like waveform
satisfies it -- beats repeat, so the phase axis is quasi-periodic. A sparse lab
trajectory does not, which is exactly why component 3 is kept separate from
components 1 and 2.

Model per beat ``b`` at normalised cardiac phase ``p`` in [0, 1):

    u[b, p] = A_b * pulse(p)                 # pulse wave + dicrotic notch
            + R_b * cos(2*pi*f_resp*t_bp)    # respiratory modulation
            + D_b                            # slow baseline drift
            + eps                            # sensor noise

with beat-to-beat amplitude ``A_b`` following an AR(1) process, respiratory
sinus arrhythmia modulating the beat period, and a smooth baseline oscillation.

How predictable the future is depends entirely on ``ar_a``
----------------------------------------------------------
With ``ar_a = 0.75`` the correlation between beat ``b`` and beat ``b+8`` is
``0.75**8 = 0.10`` -- the operator *cannot* beat the trivial zero prediction,
and should not be expected to. We therefore ship ``ar_a = 0.97``. Keep this in
mind when reading any "our operator extrapolates N steps ahead" claim: if the
future is not determined by the observed past, no architecture recovers it.

Swap this module out for a real MIMIC-IV waveform reader when you have one:
produce a ``(n_beats, n_phase)`` matrix per record and the training script needs
no other change.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _pulse(phase: np.ndarray, notch_pos: float = 0.42,
           notch_width: float = 0.05) -> np.ndarray:
    """Canonical arterial pulse: harmonics plus a dicrotic notch."""
    p = phase
    u = (1.00 * np.sin(2 * np.pi * p)
         + 0.34 * np.sin(4 * np.pi * p + 0.5)
         + 0.16 * np.sin(6 * np.pi * p + 1.0)
         + 0.07 * np.sin(8 * np.pi * p + 1.6))
    # Dicrotic notch: a small Gaussian bump riding on the downstroke.
    u += 0.22 * np.exp(-0.5 * ((p - notch_pos) / notch_width) ** 2)
    return u


def generate_waveform(n_beats: int = 24, n_phase: int = 64, fs: float = 125.0,
                      hr: float = 80.0, f_resp: float = 0.25,
                      noise: float = 0.02, seed: int | None = None,
                      ar_a: float = 0.97, ar_sigma: float = 0.05) -> np.ndarray:
    """Return a ``(n_beats, n_phase)`` beat matrix of a synthetic ABP-like signal.

    ``ar_a`` controls how much of the future is recoverable from the past;
    ``ar_sigma`` is the beat-to-beat innovation. See the module docstring.
    """
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)

    # AR(1) beat-to-beat amplitude (pulse pressure variability).
    amp = np.empty(n_beats)
    amp[0] = 1.0 + 0.10 * rng.standard_normal()
    for b in range(1, n_beats):
        amp[b] = ar_a * amp[b - 1] + ar_sigma * rng.standard_normal()

    # Smooth baseline oscillation (replaces an unpredictable random walk).
    phi = float(rng.uniform(0, 2 * np.pi))
    beat_idx = np.arange(n_beats)
    drift = 0.15 * np.sin(2 * np.pi * beat_idx / 12.0 + phi)

    out = np.empty((n_beats, n_phase), dtype=np.float32)
    t_abs = 0.0
    for b in range(n_beats):
        # Respiratory sinus arrhythmia: beat period wobbles with respiration.
        period = 60.0 / hr * (1.0 + 0.06 * np.sin(2 * np.pi * f_resp * t_abs))
        resp = 0.10 * np.cos(2 * np.pi * f_resp * (t_abs + phase * period))
        out[b] = amp[b] * _pulse(phase) + resp + drift[b]
        out[b] += noise * rng.standard_normal(n_phase)
        t_abs += period

    return out


def make_dataset(n: int, n_beats: int = 24, n_phase: int = 64,
                 n_in: int = 8, n_out: int = 8,                  seed: int = 0,
                 hr_range: Tuple[float, float] = (60.0, 110.0),
                 noise: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) for the time-advancement task.

    ``X[:, :n_in]``   = beats 0 .. n_in-1
    ``Y[:, :n_out]``  = beats n_in .. n_in+n_out-1
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, n_in, n_phase), dtype=np.float32)
    Y = np.empty((n, n_out, n_phase), dtype=np.float32)
    for i in range(n):
        hr = float(rng.uniform(*hr_range))
        f_resp = float(rng.uniform(0.15, 0.35))
        w = generate_waveform(n_beats=n_in + n_out + 8, n_phase=n_phase,
                              fs=125.0, hr=hr, f_resp=f_resp, noise=noise,
                              seed=int(rng.integers(1 << 30)))
        X[i] = w[:n_in]
        Y[i] = w[n_in:n_in + n_out]
    return X, Y


def standardise(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Global z-scoring (fit on X only). Returns scaled arrays and the stats."""
    mu, sd = float(X.mean()), float(X.std())
    sd = sd if sd > 1e-8 else 1.0
    return (X - mu) / sd, (Y - mu) / sd, mu, sd
