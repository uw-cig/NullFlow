"""Inverse problem modules."""

from NullFlow.inverse.deepinv_inpainting import DeepInvInpainting, build_deepinv_inpainting
from NullFlow.utils.config import DataConfig


def build_physics(
    data_config: DataConfig,
    *,
    device="cpu",
    dtype=None,
):
    """Builds the configured inverse-physics wrapper for the current dataset."""
    if dtype is None:
        import torch

        dtype = torch.float32

    return build_deepinv_inpainting(
        data_config=data_config,
        device=device,
        dtype=dtype,
    )

__all__ = [
    "DeepInvInpainting",
    "build_deepinv_inpainting",
    "build_physics",
]
