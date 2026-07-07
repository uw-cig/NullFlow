"""Weights and Biases scalar logging helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
import wandb
from NullFlow.utils.config import ExperimentConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRIC_KEY_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _resolve_output_dir(path_value: str) -> Path:
    """Resolves the run output directory relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _check_metric_keys(metrics: Mapping[str, float | int]) -> None:
    """Checks that scalar metric keys follow the expected naming pattern."""
    invalid_keys = [
        key
        for key in metrics
        if key == "global_step" or _METRIC_KEY_PATTERN.fullmatch(key) is None
    ]
    if invalid_keys:
        names = ", ".join(sorted(invalid_keys))
        raise ValueError(
            f"wandb scalar keys must be snake_case and must not include 'global_step': {names}."
        )


def _normalize_for_display(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Maps images from the configured range to ``[0, 1]`` for display."""
    value_min, value_max = value_range
    scale = value_max - value_min
    if scale <= 0:
        raise ValueError("value_range must be an increasing interval.")
    return ((images - value_min) / scale).clamp(0.0, 1.0)


def _prepare_images_for_display(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Converts one image batch to a wandb-compatible display tensor."""
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")
    return _normalize_for_display(images, value_range)


def _build_visual_grid(
    rows: tuple[
        tuple[str, torch.Tensor],
        tuple[str, torch.Tensor],
        tuple[str, torch.Tensor],
        tuple[str, torch.Tensor],
    ],
    *,
    num_cols: int,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Builds a 4xN visualization grid from four image batches."""
    if num_cols <= 0:
        raise ValueError("num_cols must be positive.")

    normalized_rows = []
    reference_shape = None
    for name, row in rows:
        if row.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W).")
        if row.shape[0] < num_cols:
            raise ValueError(f"{name} must contain at least num_cols samples.")
        if not torch.isfinite(row[:num_cols]).all():
            raise ValueError(f"{name} contains non-finite values before wandb logging.")
        normalized = _prepare_images_for_display(row[:num_cols].detach().cpu(), value_range)
        if reference_shape is None:
            reference_shape = normalized.shape[1:]
        elif normalized.shape[1:] != reference_shape:
            raise ValueError("all visualization tensors must share the same image shape.")
        normalized_rows.append(normalized)

    row_strips = [torch.cat(list(row), dim=2) for row in normalized_rows]
    return torch.cat(row_strips, dim=1)


class WandbLogger:
    """Wraps scalar Weights & Biases setup and logging for training."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        is_main_process: bool = True,
        wandb_module: Any = wandb,
    ) -> None:
        """Initializes the scalar logger for one experiment."""
        self.config = config
        self.is_main_process = is_main_process
        self._wandb = wandb_module
        self.enabled = (
            config.wandb.enabled
            and is_main_process
            and config.wandb.mode != "disabled"
        )
        self._initialized = False

        if not self.enabled:
            self.run = None
            return

        output_dir = _resolve_output_dir(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run = self._wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.run_name or config.experiment_name,
            tags=list(config.wandb.tags),
            mode=config.wandb.mode,
            config=config.to_dict(),
            dir=str(output_dir),
        )
        self._define_scalar_metrics()
        self._initialized = True

    def _define_scalar_metrics(self) -> None:
        """Defines the scalar metric layout for the current run."""
        self._wandb.define_metric("global_step")

        for metric_name in (
            "train_loss",
            "train_loss_imf",
            "train_loss_u",
            "train_loss_v",
            "train_aux_loss",
            "train_aux_loss_lpips",
            "train_aux_loss_convnext",
            "train_lr",
            "train_grad_norm",
            "train_validation_threshold_hits",
            "train_validation_threshold_triggered",
            "train_optimizer_adjustment_count",
            "train_data_time",
            "train_step_time",
            "val_loss",
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "val_fid",
        ):
            self._wandb.define_metric(metric_name, step_metric="global_step")

        self._wandb.define_metric("val_psnr", summary="max")
        self._wandb.define_metric("val_ssim", summary="max")
        self._wandb.define_metric("val_lpips", summary="min")
        self._wandb.define_metric("val_fid", summary="min")

    def log_scalars(
        self,
        metrics: Mapping[str, float | int],
        *,
        global_step: int,
    ) -> None:
        """Logs scalar metrics at one global step."""
        if not self.enabled:
            return
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        _check_metric_keys(metrics)

        payload = {
            "global_step": int(global_step),
            **{key: float(value) for key, value in metrics.items()},
        }
        self._wandb.log(payload, step=int(global_step))

    def log_grid(
        self,
        *,
        ground_truth: torch.Tensor,
        measurement: torch.Tensor,
        null_e: torch.Tensor,
        reconstruction: torch.Tensor,
        global_step: int,
    ) -> None:
        """Logs a fixed 4x3 visualization grid for validation samples."""
        if not self.enabled or not self.config.wandb.log_images:
            return
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")

        num_cols = self.config.evaluation.visualization_subset_size
        rows = (
            ("ground_truth", ground_truth),
            ("measurement", measurement),
            ("null_e", null_e),
            ("reconstruction", reconstruction),
        )
        grid = _build_visual_grid(
            rows,
            num_cols=num_cols,
            value_range=self.config.data.normalization_range,
        )
        caption = "rows: ground_truth, measurement, null_e, reconstruction"
        image = self._wandb.Image(
            grid,
            caption=caption,
        )
        self._wandb.log(
            {
                "global_step": int(global_step),
                "val_grid": image,
            },
            step=int(global_step),
        )

    def finish(self) -> None:
        """Finishes the active Weights & Biases run when logging is enabled."""
        if not self.enabled or not self._initialized:
            return
        self._wandb.finish()
        self._initialized = False
