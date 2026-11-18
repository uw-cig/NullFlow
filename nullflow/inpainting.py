"""Inpainting forward model  A x = M * x  with a 0/1 pixel mask shared by the colour channels.

Tensor conventions (as in mri.py, with the mask in place of the coil maps):
    image x      (B, C, H, W)      in [-1, 1]
    mask  M      (B, 1, H, W)      1 = observed pixel, 0 = missing
    measurement  y = M * x         (B, C, H, W), zero in the holes

A^H A = M is diagonal, so the regularized pseudo-inverse is elementwise:
(A^H A + lam I)^-1 A^H r = M * r / (M + lam).

Masks, drawn per image by `context`:
    box     a centred square hole of box_size pixels
    random  exactly round(drop * H * W) missing pixels, chosen uniformly by torch.randperm on the
            global CPU generator, so seeding torch fixes the mask
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor


class Inpainting:
    name = "inpainting"
    extra_cond = 1          # the mask is one extra conditioning channel: the network sees [z, A^H y, M]

    def __init__(self, mask: str = "box", box_size: int = 128, drop: float = 0.7):
        if mask not in ("box", "random"):
            raise ValueError(f"inpainting.mask must be 'box' or 'random', got {mask!r}")
        self.mask_type = mask
        self.box_size = int(box_size)
        self.drop = float(drop)

    @torch.no_grad()
    def make_mask(self, B: int, H: int, W: int) -> Tensor:
        if self.mask_type == "box":
            m = torch.ones(B, 1, H, W)
            s = min(self.box_size, H, W)
            top, left = (H - s) // 2, (W - s) // 2
            m[:, :, top:top + s, left:left + s] = 0.0
            return m
        n = H * W
        m = torch.ones(B, n)
        for i in range(B):
            m[i, torch.randperm(n)[:round(self.drop * n)]] = 0.0
        return m.view(B, 1, H, W)

    # The per-image context of this operator is the mask (B, 1, H, W).
    def context(self, batch: dict, device) -> Tensor:
        x = batch["target"]
        return self.make_mask(x.shape[0], x.shape[-2], x.shape[-1]).to(device, non_blocking=True)

    def image_shape(self, y: Tensor, mask: Tensor) -> tuple:
        return tuple(y.shape)

    def A(self, x: Tensor, mask: Tensor) -> Tensor:
        return mask * x

    def AH(self, y: Tensor, mask: Tensor) -> Tensor:
        return mask * y

    def pinv(self, r: Tensor, mask: Tensor, lam: float) -> Tensor:
        """(A^H A + lam I)^-1 A^H r."""
        return mask * (r / (mask + lam).clamp_min(1e-8))

    def observed(self, n: Tensor, mask: Tensor) -> Tensor:
        return mask * n

    def cond(self, y: Tensor, mask: Tensor) -> Tensor:
        """Network conditioning: [A^H y, M]."""
        return torch.cat([self.AH(y, mask), mask], dim=1)

    def restore(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Paste the observed pixels back into a reconstruction."""
        return mask * y + (1.0 - mask) * x
