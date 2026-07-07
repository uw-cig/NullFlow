"""Command-line solving entry point for one-step FFHQ inpainting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from NullFlow.data import build_datasets, safe_collate
from NullFlow.evaluation.inpainting import ValidationResult, run_validation
from NullFlow.inverse import build_physics
from NullFlow.model.pmf_adapter import build_pmf_model
from NullFlow.training.checkpoint import load_from_config
from NullFlow.utils.config import ExperimentConfig, load_experiment_config


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SolveResult:
    """Stores one solving pass summary and saved artifact paths."""

    checkpoint_path: Path
    metrics: dict[str, float]
    metrics_path: Path
    metadata: dict[str, object]
    metadata_path: Path
    grid_path: Path


def _default_device() -> str:
    """Returns the default runtime device for the current machine."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_output_dir(path_value: str) -> Path:
    """Resolves an output directory relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _build_val_loader(config: ExperimentConfig) -> DataLoader:
    """Builds the validation loader used by one-step solving."""
    _, dataset = build_datasets(config.data)
    use_persistent_workers = (
        config.training.persistent_workers and config.training.num_workers > 0
    )
    return DataLoader(
        dataset,
        batch_size=config.evaluation.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=use_persistent_workers,
        drop_last=False,
        collate_fn=safe_collate,
    )


def _normalize_for_display(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Maps images from the configured range to ``[0, 1]`` for saving."""
    value_min, value_max = value_range
    scale = value_max - value_min
    if scale <= 0:
        raise ValueError("value_range must be an increasing interval.")
    normalized = (images - value_min) / scale
    return normalized.clamp(0.0, 1.0)


def _build_grid(
    result: ValidationResult,
    *,
    num_cols: int,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Builds the local 4xN visualization grid from one validation result."""
    if result.visuals is None:
        raise ValueError("run_validation did not return any visualization samples.")

    rows = (
        result.visuals.ground_truth,
        result.visuals.measurement,
        result.visuals.null_e,
        result.visuals.reconstruction,
    )
    normalized_rows = []
    reference_shape = None
    for row in rows:
        if row.ndim != 4:
            raise ValueError("visualization tensors must have shape (B, C, H, W).")
        if row.shape[0] < num_cols:
            raise ValueError("visualization tensors must contain at least num_cols samples.")
        if not torch.isfinite(row[:num_cols]).all():
            raise ValueError("visualization tensors must be finite before saving.")
        if reference_shape is None:
            reference_shape = row.shape[1:]
        elif row.shape[1:] != reference_shape:
            raise ValueError("all visualization tensors must share the same image shape.")
        normalized_rows.append(
            _normalize_for_display(row[:num_cols].detach().cpu(), value_range)
        )

    row_strips = [torch.cat(list(row), dim=2) for row in normalized_rows]
    return torch.cat(row_strips, dim=1)


def _save_grid(
    result: ValidationResult,
    *,
    grid_path: Path,
    value_range: tuple[float, float],
    num_cols: int,
) -> Path:
    """Saves the first three validation samples as one 4x3 PNG grid."""
    grid = _build_grid(
        result,
        num_cols=num_cols,
        value_range=value_range,
    )
    image = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(grid_path)
    return grid_path


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for one-step solving."""
    parser = argparse.ArgumentParser(
        description="Run one-step FFHQ inpainting from a checkpoint.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the solving YAML file.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Torch device for solving, such as cpu or cuda:0.",
    )
    return parser


def run_solving(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
) -> SolveResult:
    """Loads one checkpoint, runs validation solving, and saves local artifacts."""
    if config.training.load_from is None:
        raise ValueError("training.load_from must point to the checkpoint used for solving.")

    target_device = torch.device(device)
    model = build_pmf_model(config.model, config.data, eval_mode=False).to(target_device)
    physics = build_physics(
        config.data,
        device=target_device,
        dtype=torch.float32,
    )
    resume_state = load_from_config(
        config,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=target_device,
    )
    if resume_state is None:
        raise ValueError("training.load_from must point to an existing checkpoint.")

    val_loader = _build_val_loader(config)
    result = run_validation(
        model,
        physics,
        val_loader,
        config,
        device=target_device,
    )

    solve_dir = _resolve_output_dir(config.output_dir) / "solve" / resume_state.checkpoint_path.stem
    metrics_path = solve_dir / "metrics.json"
    metadata_path = solve_dir / "metadata.json"
    grid_path = solve_dir / "grid.png"
    solve_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result.scalars, handle, indent=2, sort_keys=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2, sort_keys=True)
    _save_grid(
        result,
        grid_path=grid_path,
        value_range=config.data.normalization_range,
        num_cols=config.evaluation.visualization_subset_size,
    )

    return SolveResult(
        checkpoint_path=resume_state.checkpoint_path,
        metrics=result.scalars,
        metrics_path=metrics_path,
        metadata=result.metadata,
        metadata_path=metadata_path,
        grid_path=grid_path,
    )


def main(argv: Sequence[str] | None = None) -> SolveResult:
    """Loads config, runs solving, and prints the average metrics."""
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    result = run_solving(
        config,
        device=args.device,
    )

    print(f"checkpoint: {result.checkpoint_path}")
    print("metrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6f}")
    print(f"grid_path: {result.grid_path}")
    print(f"metrics_path: {result.metrics_path}")
    print(f"metadata_path: {result.metadata_path}")
    return result


if __name__ == "__main__":
    main()
