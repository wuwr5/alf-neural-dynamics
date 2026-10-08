"""Component 3 -- Fourier / spectral global convolution (FNO layers).

WHAT THIS IS
------------
The Fourier neural operator replaces a local kernel with a **global kernel
parameterised in the frequency domain**:

    (K v)(x) = F^{-1}{ R_k . F{v}_k }      for |k| <= k_max
               + (local 1x1 convolution)

i.e. FFT, truncate to the lowest ``k_max`` modes, mix channels with a complex
weight per mode, inverse FFT. The payoff is that the operator is
*discretisation-invariant*: train on 64 grid points, evaluate on 256.

WHY IT IS HERE AND NOT IN COMPONENTS 1-2
----------------------------------------
The spectral layer assumes the input lives on a **translation-invariant,
quasi-periodic domain**. That is true for a physiological waveform's cardiac
phase axis (beats repeat) and false for a sparse lab trajectory:

* the marker axis is an arbitrary column ordering -- permute the 19 labs and
  the spectrum changes, so the "modes" are an artefact;
* the time axis of a lab trajectory is non-periodic, non-stationary, censored,
  and interrupted by interventions.

So: use this on **dense high-frequency signals** -- MIMIC-IV waveform
(arterial line at 125 Hz, ECG, ventilator pressure/flow loops), not on labs.
Until you point it at real waveform data it runs on the synthetic generator in
``synthetic_waveform.py``, which is beat-structured on purpose so the Fourier
assumption actually holds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> keep ``modes`` -> complex channel mix -> IFFT."""

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, M)   w: (in_ch, out_ch, M)  ->  (B, out_ch, M)
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        m = min(self.modes, L // 2 + 1)
        xf = torch.fft.rfft(x, dim=-1)                     # (B, C, L//2+1)
        out = torch.zeros(B, self.out_ch, L // 2 + 1,
                          device=x.device, dtype=torch.cfloat)
        out[:, :, :m] = self.compl_mul1d(xf[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=L, dim=-1)


class SpectralConv2d(nn.Module):
    """2D Fourier layer (provided for imaging / field problems)."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                                   dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                                   dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                          device=x.device, dtype=torch.cfloat)
        out[:, :, :m1, :m2] = self.compl_mul2d(xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self.compl_mul2d(xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W))


class FNOBlock1d(nn.Module):
    """Spectral branch + pointwise bypass + nonlinearity."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.point = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.point(x))


class FNO1d(nn.Module):
    """Time-advancement operator on a beat-structured waveform.

    Input  ``(B, n_in,  L)``: ``n_in``  consecutive beats, each sampled at ``L``
    phase points.
    Output ``(B, n_out, L)``: the following ``n_out`` beats.

    The FFT runs over the **phase** axis, which is the axis that is genuinely
    quasi-periodic. Beat-to-beat dynamics are carried by the channel mixing in
    the spectral weights and the 1x1 bypass.
    """

    def __init__(self, n_in: int, n_out: int, width: int = 32, modes: int = 16,
                 n_layers: int = 4) -> None:
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.lift = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in, L) -> (B, L, width)
        h = self.lift(x.transpose(1, 2))
        h = h.transpose(1, 2)                       # (B, width, L)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)                       # (B, L, width)
        return self.proj(h).transpose(1, 2)         # (B, n_out, L)


class FNO2d(nn.Module):
    """Plain 2D FNO for spatial field / imaging problems."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, modes: int = 12,
                 n_layers: int = 4) -> None:
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList(
            [nn.ModuleList([SpectralConv2d(width, width, modes, modes),
                            nn.Conv2d(width, width, 1)]) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for spec, point in self.blocks:
            h = F.gelu(spec(h) + point(h))
        return self.proj(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
