"""Single-step training helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer

from NullFlow.utils.config import TrainingConfig


LossFn = Callable[[Any], dict[str, Any]]


@dataclass
class TrainStepResult:
    """Stores detached step outputs and scalar logs."""

    scalars: dict[str, float]
    outputs: dict[str, Any]


def _detach_tree(value: Any) -> Any:
    """Detaches tensors from nested output structures."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


def _collect_non_finite_paths(
    value: Any,
    *,
    prefix: str,
) -> list[str]:
    """Collects output paths whose tensors contain non-finite values."""
    if torch.is_tensor(value):
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            return [prefix]
        return []
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_collect_non_finite_paths(item, prefix=child_prefix))
        return paths
    if isinstance(value, tuple):
        paths: list[str] = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(_collect_non_finite_paths(item, prefix=child_prefix))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(_collect_non_finite_paths(item, prefix=child_prefix))
        return paths
    return []


def _raise_if_non_finite_outputs(outputs: dict[str, Any]) -> None:
    """Raises a clear error when one training output contains NaN or Inf."""
    bad_paths = _collect_non_finite_paths(outputs, prefix="")
    if not bad_paths:
        return
    joined = ", ".join(bad_paths)
    raise ValueError(f"loss_fn produced non-finite tensors at: {joined}.")


def _get_lr(optimizer: Optimizer) -> float:
    """Returns the learning rate from the first optimizer parameter group."""
    if not optimizer.param_groups:
        raise ValueError("optimizer must contain at least one parameter group.")
    return float(optimizer.param_groups[0]["lr"])


def _extract_train_scalars(outputs: dict[str, Any]) -> dict[str, float]:
    """Builds train scalar logs from scalar tensors returned by ``loss_fn``."""
    scalar_map: dict[str, float] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim != 0:
            continue
        if not torch.is_floating_point(value):
            continue
        if key == "loss":
            scalar_map["train_loss"] = float(value.detach().cpu())
            continue
        scalar_map[f"train_{key}"] = float(value.detach().cpu())
    return scalar_map


def _get_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Computes the global L2 norm of available gradients."""
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = [gradient.norm(2) for gradient in grads]
    total_norm = torch.stack(norms).norm(2)
    return float(total_norm.detach().cpu())


def _clip_or_measure_grad_norm(
    model: nn.Module,
    max_norm: float | None,
) -> float:
    """Clips gradients when requested and returns the pre-clip global norm."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if max_norm is None:
        return _get_grad_norm(parameters)

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return float(grad_norm.detach().cpu())


def _as_minibatches(
    x: Any,
) -> list[Any]:
    """Normalizes one batch or a batch sequence into a non-empty list."""
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, Mapping):
        return [x]
    minibatches = list(x)
    if not minibatches:
        raise ValueError("run_train_step requires at least one minibatch.")
    return minibatches


def _extract_batch_tensor(batch: Any) -> torch.Tensor:
    """Returns the training image tensor from one batch object."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        x = batch.get("x")
        if not torch.is_tensor(x):
            raise ValueError("mapping minibatches must contain a tensor 'x'.")
        return x
    raise ValueError("minibatches must be tensors or mappings containing 'x'.")


def _move_batch_to_device(batch: Any, target_device: torch.device) -> Any:
    """Moves all tensors inside one batch object to the requested device."""
    if torch.is_tensor(batch):
        return batch.to(device=target_device)
    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(value, target_device)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(item, target_device) for item in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(item, target_device) for item in batch]
    return batch


def _merge_outputs(
    current: Any,
    update: Any,
    *,
    chunk_size: int,
    total_size: int,
) -> Any:
    """Merges detached minibatch outputs into one step-level structure."""
    if torch.is_tensor(update):
        detached = update.detach().cpu()
        if detached.ndim == 0:
            weighted = detached * (chunk_size / total_size)
            return weighted if current is None else current + weighted
        # Skip accumulating large image/k-space tensors (ndim >= 4) to avoid
        # exhausting CPU RAM when many minibatches are accumulated.
        if detached.ndim >= 4:
            return None
        if detached.shape[0] == chunk_size:
            if current is None:
                return detached
            if not torch.is_tensor(current):
                return current
            if current.ndim != detached.ndim:
                return current
            if current.shape[1:] != detached.shape[1:]:
                current, detached = _pad_batch_tensors_for_concat(current, detached)
            return torch.cat((current, detached), dim=0)
        return detached if current is None else current
    if isinstance(update, dict):
        merged = {} if current is None else dict(current)
        for key, value in update.items():
            merged[key] = _merge_outputs(
                merged.get(key),
                value,
                chunk_size=chunk_size,
                total_size=total_size,
            )
        return merged
    if isinstance(update, tuple):
        previous = current if current is not None else (None,) * len(update)
        return tuple(
            _merge_outputs(
                previous[index],
                value,
                chunk_size=chunk_size,
                total_size=total_size,
            )
            for index, value in enumerate(update)
        )
    if isinstance(update, list):
        previous = current if current is not None else [None] * len(update)
        return [
            _merge_outputs(
                previous[index],
                value,
                chunk_size=chunk_size,
                total_size=total_size,
            )
            for index, value in enumerate(update)
        ]
    return update if current is None else current


def _pad_batch_tensors_for_concat(
    current: torch.Tensor,
    update: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pads non-batch dimensions so two minibatch tensors can be concatenated."""
    if current.ndim != update.ndim:
        raise ValueError("current and update tensors must have the same rank.")
    if current.ndim == 0:
        raise ValueError("Scalar tensors do not support batch concatenation padding.")

    target_shape = (
        0,
        *tuple(
            max(int(current_size), int(update_size))
            for current_size, update_size in zip(current.shape[1:], update.shape[1:], strict=True)
        ),
    )
    return (
        _pad_non_batch_dimensions(current, target_shape),
        _pad_non_batch_dimensions(update, target_shape),
    )


def _pad_non_batch_dimensions(
    tensor: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """Zero-pads one tensor across non-batch dimensions to one target shape."""
    if tensor.ndim != len(target_shape):
        raise ValueError("target_shape must match tensor.ndim.")
    if tensor.ndim == 0:
        raise ValueError("Scalar tensors do not support non-batch padding.")
    if any(
        int(current_size) > int(target_size)
        for current_size, target_size in zip(tensor.shape[1:], target_shape[1:], strict=True)
    ):
        raise ValueError("target_shape must be at least as large as tensor.shape in non-batch dimensions.")
    if tensor.shape[1:] == target_shape[1:]:
        return tensor.contiguous()

    padded = torch.zeros(
        (tensor.shape[0], *target_shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    slices = (slice(None),) + tuple(slice(0, int(size)) for size in tensor.shape[1:])
    padded[slices] = tensor
    return padded


def run_train_step(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: LossFn,
    x: Any,
    *,
    training_config: TrainingConfig,
    device: torch.device | str | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainStepResult:
    """Runs one optimizer step from one or more minibatches.

    Args:
        model: Model updated by the training step.
        optimizer: Optimizer used for the parameter update.
        loss_fn: Callable that returns a loss dictionary containing ``loss``.
        x: One minibatch tensor or a sequence of minibatch tensors.
        training_config: Training options including AMP and grad clipping.
        device: Optional target device used to move each minibatch before the forward pass.
        scaler: Optional gradient scaler used for CUDA mixed precision.

    Returns:
        Detached outputs from ``loss_fn`` plus scalar logs for the step.
    """
    minibatches = _as_minibatches(x)
    total_batch_size = 0
    for minibatch in minibatches:
        minibatch_tensor = _extract_batch_tensor(minibatch)
        if minibatch_tensor.ndim != 4:
            raise ValueError("x must have shape (B, C, H, W).")
        if not torch.is_floating_point(minibatch_tensor):
            raise ValueError("x must be a floating-point tensor.")
        total_batch_size += int(minibatch_tensor.shape[0])
    target_device = None if device is None else torch.device(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    merged_outputs = None
    merged_scalars: dict[str, float] = {}
    grad_device_type = None

    for minibatch in minibatches:
        if target_device is not None:
            minibatch = _move_batch_to_device(minibatch, target_device)
        minibatch_tensor = _extract_batch_tensor(minibatch)
        if grad_device_type is None:
            grad_device_type = minibatch_tensor.device.type

        use_amp = (
            training_config.amp_enabled
            and scaler is not None
            and minibatch_tensor.device.type == "cuda"
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )

        with amp_context:
            outputs = loss_fn(minibatch)

        if not isinstance(outputs, dict):
            raise ValueError("loss_fn must return a dictionary of step outputs.")
        _raise_if_non_finite_outputs(outputs)

        loss = outputs.get("loss")
        if not torch.is_tensor(loss):
            raise ValueError("loss_fn must return a dictionary containing a tensor 'loss'.")
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor under the 'loss' key.")

        chunk_size = int(minibatch_tensor.shape[0])
        loss_weight = chunk_size / total_batch_size
        output_scalars = _extract_train_scalars(outputs)
        for key, value in output_scalars.items():
            merged_scalars[key] = merged_scalars.get(key, 0.0) + value * loss_weight
        merged_outputs = _merge_outputs(
            merged_outputs,
            outputs,
            chunk_size=chunk_size,
            total_size=total_batch_size,
        )
        scaled_loss = loss * loss_weight

        if use_amp:
            assert scaler is not None
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    use_amp = (
        training_config.amp_enabled
        and scaler is not None
        and grad_device_type == "cuda"
    )
    if use_amp:
        assert scaler is not None
        scaler.unscale_(optimizer)
        grad_norm = _clip_or_measure_grad_norm(model, training_config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = _clip_or_measure_grad_norm(model, training_config.grad_clip_norm)
        optimizer.step()

    scalars = dict(merged_scalars)
    scalars["train_lr"] = _get_lr(optimizer)
    scalars["train_grad_norm"] = grad_norm
    return TrainStepResult(
        scalars=scalars,
        outputs=_detach_tree(merged_outputs),
    )
