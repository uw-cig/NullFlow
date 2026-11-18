"""Image-quality metrics.

MRI, on the magnitude image with each slice's own dynamic range (psnr_ssim, LPIPS.__call__):
PSNR / SSIM:  a = |x_hat|, b = |x_gt|, P = max b;  a is clipped to [0, P];
              PSNR = 10 log10(P^2 / mean (a - b)^2);  SSIM = torchmetrics (Gaussian 11, sigma 1.5), data range P.
LPIPS:        AlexNet (lpips 0.1) on a / P and b / P clamped to [0, 1], grey replicated to 3 channels,
              mapped to [-1, 1].

RGB images in [-1, 1] (rgb_psnr_ssim, LPIPS.rgb):
PSNR / SSIM:  on (x + 1) / 2 clamped to [0, 1], data range 1; SSIM = torchmetrics (Gaussian 11, sigma 1.5).
LPIPS:        AlexNet on the images clamped to [-1, 1].
"""
from __future__ import annotations

import torch
from torchmetrics.functional import structural_similarity_index_measure


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """(2, H, W) real / imag -> (H, W) magnitude."""
    return torch.view_as_complex(x.permute(1, 2, 0).contiguous()).abs()


@torch.no_grad()
def psnr_ssim(x_hat: torch.Tensor, x_gt: torch.Tensor) -> tuple:
    """x_hat, x_gt: (2, H, W). Returns (PSNR in dB, SSIM)."""
    a = magnitude(x_hat.to(torch.float64))
    b = magnitude(x_gt.to(torch.float64))
    P = b.max()
    a = a.clip(0, P)
    psnr = 10.0 * torch.log10(P ** 2 / (torch.mean((a - b) ** 2) + 1e-20))
    ssim = structural_similarity_index_measure(a[None, None].float(), b[None, None].float(),
                                               data_range=float(P))
    return float(psnr), float(ssim)


@torch.no_grad()
def rgb_psnr_ssim(x_hat: torch.Tensor, x_gt: torch.Tensor) -> tuple:
    """x_hat, x_gt: (C, H, W) in [-1, 1]. Returns (PSNR in dB, SSIM)."""
    a = ((x_hat.float() + 1.0) / 2.0).clamp(0.0, 1.0)[None]
    b = ((x_gt.float() + 1.0) / 2.0).clamp(0.0, 1.0)[None]
    psnr = 10.0 * torch.log10(1.0 / (torch.mean((a - b) ** 2) + 1e-12))
    return float(psnr), float(structural_similarity_index_measure(a, b, data_range=1.0))


class LPIPS:
    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
        a = magnitude(x_hat.float()).to(self.device)
        b = magnitude(x_gt.float()).to(self.device)
        P = b.abs().max().clamp_min(1e-12)
        a = (a / P).clamp(0, 1)[None, None].expand(-1, 3, -1, -1)
        b = (b / P).clamp(0, 1)[None, None].expand(-1, 3, -1, -1)
        return float(self.net(2 * a - 1, 2 * b - 1).reshape(()))

    @torch.no_grad()
    def rgb(self, x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
        """x_hat, x_gt: (3, H, W) RGB images in [-1, 1]."""
        a = x_hat.float().clamp(-1, 1)[None].to(self.device)
        b = x_gt.float().clamp(-1, 1)[None].to(self.device)
        return float(self.net(a, b).reshape(()))
