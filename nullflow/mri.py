"""Multi-coil Cartesian MRI forward model  A x = M F (S x).

Tensor conventions (kept throughout the package):
    image x      (B, 2, H, W)      real / imaginary channels of the complex image
    maps  S      (B, C, H, W)      complex coil sensitivities, zero-padded to C coils
    k-space y    (B, C, H, W, 2)   real view of the masked multi-coil k-space
    mask  M      (1, 1, 1, W)      Cartesian phase-encode mask, shared by every slice

A padded coil has S_c = 0, so it contributes nothing to A or A^H.
"""
from __future__ import annotations

import numpy as np
import torch

Tensor = torch.Tensor


def cartesian_mask(total_lines: int, R: int, seed: int = 0, pattern: str = "random") -> np.ndarray:
    """1-D phase-encode mask: a fully sampled centre (8% of the lines for R <= 6, else 4%)
    plus floor(total_lines / R) - #centre lines drawn from the rest."""
    acs = int(np.floor((0.08 if 1 < R <= 6 else 0.04) * total_lines))
    rng = np.random.RandomState(seed)
    n_sampled = int(np.floor(total_lines / R))
    center = np.arange((total_lines - acs) // 2, (total_lines + acs) // 2)
    outer = np.setdiff1d(np.arange(total_lines), center)
    if pattern == "random":
        picked = rng.choice(outer, size=n_sampled - acs, replace=False)
    elif pattern == "equispaced":
        picked = outer[::R]
    else:
        raise ValueError(f"unknown mask pattern {pattern!r}")
    m = np.zeros(total_lines)
    m[center] = 1.0
    m[picked] = 1.0
    return m


def fft2c(x: Tensor) -> Tensor:
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return torch.fft.ifftshift(x, dim=(-2, -1))


def ifft2c(x: Tensor) -> Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def to_complex(x: Tensor) -> Tensor:
    """(B, 2, H, W) real -> (B, H, W) complex."""
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())


def to_real(x: Tensor) -> Tensor:
    """(B, H, W) complex -> (B, 2, H, W) real."""
    return torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()


class MultiCoilMRI:
    """A = M F S with a fixed Cartesian mask (vertical phase-encode lines).

    `pinv` solves (A^H A + lam I) x = A^H y by conjugate gradient, per sample. CG runs on the
    image-space normal equations: their spectrum lies in [lam, ~1] and A^H y is already in
    range(A^H), so the zero-padded coils and the unsampled k-space lines never enter the solve.
    """

    name = "mri"
    extra_cond = 0          # the network sees [z, A^H y]: no conditioning channel beyond A^H y
    weight_norm = "fixed"   # loss weight w = (m + c)^-p

    def __init__(self, acceleration: int, total_lines: int = 320, mask_seed: int = 0,
                 pattern: str = "random", cg_iters: int = 200, cg_tol: float = 1e-6):
        self.R = int(acceleration)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        m = cartesian_mask(int(total_lines), self.R, int(mask_seed), pattern)
        self._mask = torch.from_numpy(m[None, None, None, :].copy().astype(bool))
        self._mask_dev = {}

    def mask(self, device, dtype=torch.float32) -> Tensor:
        key = (str(device), dtype)
        if key not in self._mask_dev:
            self._mask_dev[key] = self._mask.to(device).to(dtype)
        return self._mask_dev[key]

    # The per-slice context of this operator is the coil maps (B, C, H, W).
    def context(self, batch: dict, device) -> Tensor:
        return batch["maps"].to(device, non_blocking=True)

    def image_shape(self, y: Tensor, maps: Tensor) -> tuple:
        return (maps.shape[0], 2, maps.shape[-2], maps.shape[-1])

    def observed(self, k: Tensor, maps: Tensor) -> Tensor:
        """k-space restricted to the sampled lines (where measurement noise lives)."""
        return k * self.mask(k.device).unsqueeze(-1)

    def cond(self, y: Tensor, maps: Tensor) -> Tensor:
        """Network conditioning: the zero-filled adjoint A^H y."""
        return self.AH(y, maps)

    def A(self, x: Tensor, maps: Tensor) -> Tensor:
        """(B, 2, H, W) image -> (B, C, H, W, 2) masked multi-coil k-space."""
        coils = maps * to_complex(x).unsqueeze(1)
        return torch.view_as_real(self.mask(x.device) * fft2c(coils))

    def AH(self, y: Tensor, maps: Tensor) -> Tensor:
        """(B, C, H, W, 2) k-space -> (B, 2, H, W) image:  A^H y = sum_c conj(S_c) F^-1 (M y_c)."""
        yc = torch.view_as_complex(y.contiguous())
        img = ifft2c(self.mask(y.device) * yc) * torch.conj(maps)
        return to_real(img.sum(dim=1))

    @torch.no_grad()
    def pinv(self, y: Tensor, maps: Tensor, lam: float) -> Tensor:
        """(A^H A + lam I)^-1 A^H y by CG from x = 0: at most cg_iters iterations, stopping once
        every sample in the batch has ||r|| / ||b|| < cg_tol."""
        b = self.AH(y, maps)

        def normal(v):
            return self.AH(self.A(v, maps), maps) + lam * v

        def dot(u, v):
            return (u * v).flatten(1).sum(1).view(-1, 1, 1, 1)

        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = dot(r, r)
        b_norm = dot(b, b).clamp_min(1e-30)
        for _ in range(self.cg_iters):
            Ap = normal(p)
            alpha = rs / dot(p, Ap).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = dot(r, r)
            if bool(((rs_new / b_norm) < self.cg_tol ** 2).all()):
                break
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            rs = rs_new
        return x
