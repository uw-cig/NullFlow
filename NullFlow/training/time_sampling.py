"""Time sampling helpers for flow training."""

from __future__ import annotations

import torch


DEFAULT_MIN_T = 1e-3


def sample_t_r(
    batch_size: int,
    device: torch.device | str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    min_t: float = DEFAULT_MIN_T,
    max_t: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples paired time values that satisfy ``0 <= r < t <= 1``.

    Args:
        batch_size: Number of ``(t, r)`` pairs to sample.
        device: Target device for the sampled tensors.
        dtype: Target dtype for the sampled tensors.
        min_t: Lower bound for ``t``. This keeps ``t`` away from zero.
        max_t: Upper bound for ``t``.
        generator: Optional torch random generator.

    Returns:
        A tuple ``(t, r)`` with both tensors of shape ``(batch_size,)``.

    Raises:
        ValueError: If the sampling bounds are invalid.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if min_t <= 0.0:
        raise ValueError("min_t must be positive.")
    if max_t > 1.0:
        raise ValueError("max_t must not exceed 1.0.")
    if min_t >= max_t:
        raise ValueError("min_t must be smaller than max_t.")

    t = torch.rand(
        batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    t = t * (max_t - min_t) + min_t

    r = torch.rand(
        batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    r = r * t
    return t, r
