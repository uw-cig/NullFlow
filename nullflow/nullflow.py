"""NullFlow: one-step MeanFlow transport from a null-space source.

Operator. A is multi-coil MRI (mri.py; per-slice context = coil maps) or inpainting (inpainting.py;
per-image context = pixel mask). Each operator provides A(x, ctx), AH(y, ctx), the regularized
pseudo-inverse pinv(r, ctx, lam) = (A^H A + lam I)^-1 A^H r, and cond(y, ctx), the measurement
conditioning of the network: A^H y for MRI, [A^H y, mask] for inpainting.

Source. For a measurement y = A x0 (+ n), draw x_p ~ N(0, tau^2 I) and set

    x1 = x_p + (A^H A + lam I)^-1 A^H (y - A x_p - n'),   n' ~ N(0, sigma^2 I),  lam = max(sigma^2 / tau^2, 1e-8).

With sigma = 0 this is  x1 = A^+ y + P_null x_p : the range-space component of x1 is fixed by y and
only the null-space component is random. (For sigma > 0 it is the randomize-then-optimize sample of
the Gaussian posterior under the prior N(0, tau^2 I).)

Transport. Path z_t = (1 - t) x0 + t x1 (t = 0 clean, t = 1 source), velocity v = x1 - x0. The
network predicts x0 (x-prediction), u_{r,t}(z) = (z - net([z, cond(y)], r, t, sigma)) / t, and is
trained with the MeanFlow identity  u + (t - r) sg(du/dt) ~ v  (du/dt by a forward-mode JVP along
(v, 0, 1)) under the adaptive weight w = 1 / (m / m_ref + c), m the per-sample squared residual and
m_ref = 1 (weight_norm "fixed") or a running mean of m (weight_norm "ema").

Sampling is one network call:  x0_hat = net([x1, cond(y)], r = 0, t = 1, sigma).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.func import jvp


class NullFlow(torch.nn.Module):

    def __init__(self, net: torch.nn.Module, op, tau: float = 1.0, sigma: float = 0.0,
                 p_ratio: float = 0.5, corner_frac: float = 0.1, logit_mu: float = 0.0,
                 logit_sigma: float = 1.0, weight_c: float = 1e-3, weight_p: float = 1.0,
                 t_eps: float = 1e-3, weight_norm: str = "fixed", weight_norm_decay: float = 0.99):
        super().__init__()
        self.net = net
        self.op = op
        self.tau = float(tau)
        self.sigma = float(sigma)            # measurement noise std (0 for the noiseless setting)
        self.p_ratio = float(p_ratio)        # fraction of samples with r = t
        self.corner_frac = float(corner_frac)  # of the r < t samples, the share moved to r ~ 0, t ~ 1
        self.logit_mu = float(logit_mu)
        self.logit_sigma = float(logit_sigma)
        self.weight_c = float(weight_c)
        self.weight_p = float(weight_p)
        self.t_eps = float(t_eps)
        if weight_norm not in ("fixed", "ema"):
            raise ValueError(f"weight_norm must be 'fixed' or 'ema', got {weight_norm!r}")
        self.weight_norm = weight_norm
        self.weight_norm_decay = float(weight_norm_decay)   # per call of loss()
        self._m_ema = None

    # ------------------------------------------------------------------ measurement side
    def measure(self, x0: Tensor, ctx: Tensor) -> Tensor:
        """y = A x0 (+ sigma n on the observed entries)."""
        y = self.op.A(x0, ctx)
        if self.sigma > 0:
            y = y + self.sigma * self.op.observed(torch.randn_like(y), ctx)
        return y

    def cond(self, y: Tensor, ctx: Tensor) -> Tensor:
        """Conditioning image fed to the network next to z."""
        return self.op.cond(y, ctx)

    @torch.no_grad()
    def source(self, y: Tensor, ctx: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """x1 = x_p + (A^H A + lam I)^-1 A^H (y - A x_p - n')."""
        lam = max(self.sigma ** 2 / self.tau ** 2, 1e-8)
        shape = self.op.image_shape(y, ctx)
        x_p = self.tau * torch.randn(shape, device=y.device, dtype=y.dtype, generator=generator)
        n_p = self.sigma * torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
        return x_p + self.op.pinv(y - self.op.A(x_p, ctx) - n_p, ctx, lam)

    def _sigma_vec(self, B: int, device) -> Tensor:
        return torch.full((B,), self.sigma, device=device)

    # ------------------------------------------------------------------ training
    def _sample_t_r(self, B: int, device):
        """t ~ logit-normal; r = t for p_ratio of the batch, else r ~ U(0, t); of the r < t
        samples, corner_frac are moved to (r ~ U(0, 0.1), t ~ U(0.9, 1)), the corner that
        one-step sampling evaluates."""
        t = torch.sigmoid(self.logit_mu + self.logit_sigma * torch.randn(B, device=device))
        r = t.clone()
        neq = torch.rand(B, device=device) > self.p_ratio
        r = torch.where(neq, torch.rand(B, device=device) * t, r)
        if self.corner_frac > 0.0:
            corner = neq & (torch.rand(B, device=device) < self.corner_frac)
            if corner.any():
                t = torch.where(corner, 0.9 + 0.1 * torch.rand(B, device=device), t)
                r = torch.where(corner, 0.1 * torch.rand(B, device=device), r)
        return t, r

    def _u(self, z, r, t, cond, sigma):
        x0_pred = self.net(torch.cat([z, cond], dim=1), t, r, sigma)
        return (z - x0_pred) / t.view(-1, 1, 1, 1).clamp_min(self.t_eps)

    @torch.no_grad()
    def _m_ref(self, m: Tensor):
        """Normalizer of the residual in the adaptive weight: 1, or a running mean of m that is
        averaged over the GPUs so that every rank weights its samples the same way."""
        if self.weight_norm == "fixed":
            return 1.0
        bm = m.mean()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bm)
            bm = bm / dist.get_world_size()
        if self._m_ema is None:
            self._m_ema = bm.clone()
        else:
            self._m_ema.mul_(self.weight_norm_decay).add_(bm, alpha=1.0 - self.weight_norm_decay)
        return self._m_ema.clamp_min(1e-12)

    def loss(self, x0: Tensor, ctx: Tensor):
        """MeanFlow loss on a batch of clean images x0 with their operator context (coil maps or
        masks). Returns (loss, per-sample squared residual m)."""
        B = x0.shape[0]
        y = self.measure(x0, ctx)
        x1 = self.source(y, ctx)
        cond = self.cond(y, ctx)
        sigma = self._sigma_vec(B, x0.device)

        t, r = self._sample_t_r(B, x0.device)
        tm = t.view(-1, 1, 1, 1)
        z = (1.0 - tm) * x0 + tm * x1
        v = x1 - x0
        with torch.autocast(device_type="cuda", enabled=False):      # the JVP runs in fp32
            u, dudt = jvp(lambda z_, r_, t_: self._u(z_, r_, t_, cond, sigma),
                          (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
            V = u + (t - r).view(-1, 1, 1, 1) * dudt.detach()
            m = (V - v).pow(2).flatten(1).mean(dim=1)
            w = (m.detach() / self._m_ref(m) + self.weight_c) ** (-self.weight_p)
            loss = (w * m).mean()
        return loss, m.detach()

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def one_step(self, x1: Tensor, cond: Tensor) -> Tensor:
        B = x1.shape[0]
        t1 = torch.ones(B, device=x1.device)
        r0 = torch.zeros(B, device=x1.device)
        return self.net(torch.cat([x1, cond], dim=1), t1, r0, self._sigma_vec(B, x1.device))

    @torch.no_grad()
    def sample_mean(self, y: Tensor, ctx: Tensor, M_list, generator: Optional[torch.Generator] = None,
                    chunk: int = 8, net_batch: Optional[int] = None) -> dict:
        """One item (y, ctx of batch size 1): M draws x0_hat^(1..M), returned as running means
        {M: mean of the first M draws} for every M in M_list. The sources are drawn `chunk` at a
        time (one generator call each) and go through the network `net_batch` (default chunk) at a time."""
        want = sorted(set(int(m) for m in M_list))
        N = want[-1]
        net_batch = int(net_batch or chunk)
        cond = self.cond(y, ctx)
        out, run, seen = {}, None, 0
        while seen < N:
            b = min(chunk, N - seen)
            yb = y.expand(b, *y.shape[1:]).contiguous()
            cb = ctx.expand(b, *ctx.shape[1:]).contiguous()
            x1 = self.source(yb, cb, generator)
            for s in range(0, b, net_batch):
                x1s = x1[s:s + net_batch]
                x0 = self.one_step(x1s, cond.expand(x1s.shape[0], *cond.shape[1:]))
                for j in range(x0.shape[0]):
                    run = x0[j:j + 1].clone() if run is None else run + x0[j:j + 1]
                    seen += 1
                    if seen in want:
                        out[seen] = run / seen
        return out
