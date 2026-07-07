"""Custom collation helpers that avoid shared-storage resize issues."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


def _pad_first_dimension(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Zero-pads one tensor along the leading dimension to a target length."""
    if tensor.ndim == 0:
        raise ValueError("Cannot pad a scalar tensor along the first dimension.")
    if tensor.shape[0] > target_length:
        raise ValueError("target_length must be at least the current first dimension.")
    if tensor.shape[0] == target_length:
        return tensor.contiguous()

    padded = torch.zeros(
        (target_length, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[: tensor.shape[0]] = tensor
    return padded


def _stack_tensor_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stacks one tensor batch, padding the leading dimension when coil counts vary."""
    tensors = [item.contiguous() for item in batch]
    reference_shape = tensors[0].shape
    if all(tensor.shape == reference_shape for tensor in tensors):
        return torch.stack(tensors, dim=0)

    if all(tensor.ndim == tensors[0].ndim for tensor in tensors) and all(
        tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors
    ):
        target_length = max(int(tensor.shape[0]) for tensor in tensors)
        padded = [_pad_first_dimension(tensor, target_length) for tensor in tensors]
        return torch.stack(padded, dim=0)

    shape_summary = ", ".join(str(tuple(tensor.shape)) for tensor in tensors)
    raise RuntimeError(
        "safe_collate received tensors with incompatible shapes: "
        f"{shape_summary}"
    )


def safe_collate(batch):
    """Recursively collates samples without using tensor shared-storage fast paths."""
    if not batch:
        raise ValueError("batch must contain at least one sample.")

    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return _stack_tensor_batch(batch)
    if isinstance(elem, Mapping):
        return {
            key: safe_collate([sample[key] for sample in batch])
            for key in elem
        }
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(safe_collate(list(items)) for items in zip(*batch, strict=True)))
    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        transposed = zip(*batch, strict=True)
        return [safe_collate(list(items)) for items in transposed]
    return default_collate(batch)


__all__ = ["safe_collate"]
