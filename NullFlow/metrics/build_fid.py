"""Builds cached FID reference statistics from a configured dataset split."""

from __future__ import annotations

import argparse
from dataclasses import replace

from NullFlow.data import build_datasets
from NullFlow.metrics.reconstruction import build_fid_stats
from NullFlow.utils.config import load_experiment_config


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for FID stats generation."""
    parser = argparse.ArgumentParser(
        description="Build cached FID reference statistics for a dataset split.",
    )
    parser.add_argument(
        "--config",
        default="configs/ffhq256_inpainting.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Dataset split used to build the reference statistics.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output .npz path. Defaults to evaluation.fid_real_stats_path for val.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override used while extracting features.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers used while building the stats.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for the feature extractor.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the FID build progress bar.",
    )
    return parser.parse_args()


def _default_out_path(config, split: str) -> str:
    """Builds the default output path for one split."""
    if split == "val" and config.evaluation.fid_real_stats_path is not None:
        return config.evaluation.fid_real_stats_path
    return f"fid_stats/{config.data.dataset_name}{config.data.image_size}_{split}.npz"


def main() -> None:
    """Builds and saves cached FID statistics from the configured dataset."""
    args = _parse_args()
    config = load_experiment_config(args.config)
    data_config = replace(config.data, enable_random_flip=False)
    train_dataset, val_dataset = build_datasets(data_config)
    dataset = train_dataset if args.split == "train" else val_dataset

    out_path = args.out or _default_out_path(config, args.split)
    batch_size = args.batch_size or config.evaluation.val_batch_size
    saved_path = build_fid_stats(
        dataset,
        out_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=args.device,
        value_range=data_config.normalization_range,
        show_progress=not args.quiet,
    )
    print(saved_path)


if __name__ == "__main__":
    main()
