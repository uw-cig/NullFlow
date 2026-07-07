"""Checkpoint save and restore helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import uuid
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from NullFlow.utils.config import ExperimentConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ResumeState:
    """Stores the state returned after loading a checkpoint."""

    checkpoint_path: Path
    epoch: int
    global_step: int
    config: dict[str, Any]
    engine_state: dict[str, Any] = field(default_factory=dict)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Returns the underlying model when wrapped by DDP-like containers."""
    wrapped_model = getattr(model, "module", None)
    if isinstance(wrapped_model, nn.Module):
        return wrapped_model
    return model


def _resolve_path(path_value: str | Path) -> Path:
    """Resolves a checkpoint path relative to the repository root when needed."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def build_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | Any | None,
    scheduler: LRScheduler | Any | None,
    *,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
    engine_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the serialized checkpoint payload."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    if global_step < 0:
        raise ValueError("global_step must be non-negative.")

    return {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config.to_dict(),
        "engine_state": {} if engine_state is None else dict(engine_state),
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | Any | None,
    scheduler: LRScheduler | Any | None,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
    engine_state: dict[str, Any] | None = None,
) -> Path:
    """Saves a checkpoint file and returns the resolved output path."""
    resolved_path = _resolve_path(checkpoint_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        config=config,
        engine_state=engine_state,
    )
    temp_path = resolved_path.parent / f".{resolved_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        try:
            torch.save(payload, temp_path)
        except RuntimeError as error:
            message = str(error)
            if "unexpected pos" not in message and "iostream error" not in message:
                raise
            torch.save(
                payload,
                temp_path,
                _use_new_zipfile_serialization=False,
            )
        os.replace(temp_path, resolved_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return resolved_path


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | Any | None = None,
    scheduler: LRScheduler | Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> ResumeState:
    """Loads a checkpoint into the provided objects and returns resume metadata."""
    resolved_path = _resolve_path(checkpoint_path)
    payload = torch.load(
        resolved_path,
        map_location=map_location,
        weights_only=False,
    )

    if "model" not in payload:
        raise ValueError("Checkpoint is missing the 'model' state.")

    _unwrap_model(model).load_state_dict(payload["model"], strict=strict)

    optimizer_state = payload.get("optimizer")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scaler_state = payload.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    scheduler_state = payload.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    return ResumeState(
        checkpoint_path=resolved_path,
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        config=dict(payload.get("config", {})),
        engine_state=dict(payload.get("engine_state", {})),
    )


def load_from_config(
    config: ExperimentConfig,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | Any | None = None,
    scheduler: LRScheduler | Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> ResumeState | None:
    """Loads the checkpoint configured under ``training.load_from`` when present."""
    checkpoint_path = config.training.load_from
    if checkpoint_path is None:
        return None
    return load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        map_location=map_location,
        strict=strict,
    )
