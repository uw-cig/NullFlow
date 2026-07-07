"""FFHQ dataset access and fixed split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from NullFlow.data.transforms import build_train_transform, build_val_transform
from NullFlow.utils.config import DataConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
Split = Literal["train", "val", "test"]


def _resolve_repo_path(path_value: str) -> Path:
    """Resolves a config path relative to the repository root when needed.

    Args:
        path_value: Configured file-system path.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path value is empty.
    """
    if not path_value:
        raise ValueError("Configured dataset path must be non-empty.")

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _resolve_image_root(data_config: DataConfig) -> Path:
    """Selects the image directory for FFHQ samples.

    Args:
        data_config: Dataset configuration.

    Returns:
        The directory that stores FFHQ image files.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
        ValueError: If resized caching is enabled without a cache directory.
    """
    if data_config.cache_resized_images:
        if data_config.resized_cache_dir is None:
            raise ValueError(
                "resized_cache_dir must be set when cache_resized_images is enabled."
            )
        image_root = _resolve_repo_path(data_config.resized_cache_dir)
    else:
        image_root = _resolve_repo_path(data_config.data_root)

    if not image_root.is_dir():
        raise FileNotFoundError(f"FFHQ image directory does not exist: {image_root}")
    return image_root


def _get_split_bounds(data_config: DataConfig, split: Split) -> tuple[int, int]:
    """Returns the configured inclusive-exclusive bounds for a split."""
    if split == "train":
        return data_config.train_split_start, data_config.train_split_end
    if split == "test":
        return data_config.test_split_start, data_config.test_split_end
    return data_config.val_split_start, data_config.val_split_end


def _discover_image_paths(image_root: Path) -> dict[int, Path]:
    """Indexes flat FFHQ image files by their numeric filename stem.

    Args:
        image_root: Directory containing FFHQ images such as ``00000.png``.

    Returns:
        A mapping from integer sample index to image path.

    Raises:
        ValueError: If duplicate numeric filenames are found.
    """
    indexed_paths: dict[int, Path] = {}
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in _VALID_IMAGE_SUFFIXES:
            continue
        if not image_path.stem.isdigit():
            continue

        image_index = int(image_path.stem)
        if image_index in indexed_paths:
            raise ValueError(
                f"Duplicate FFHQ index {image_index:05d} found in {image_root}."
            )
        indexed_paths[image_index] = image_path
    return indexed_paths


def _build_split_image_paths(
    image_root: Path,
    split_start: int,
    split_end: int,
) -> list[Path]:
    """Builds the ordered image list for one configured split.

    Args:
        image_root: Directory containing FFHQ image files.
        split_start: First numeric FFHQ index in the split.
        split_end: Exclusive numeric FFHQ index after the split.

    Returns:
        Ordered image paths for the requested split.

    Raises:
        FileNotFoundError: If one or more expected FFHQ files are missing.
    """
    indexed_paths = _discover_image_paths(image_root)
    missing_indices = [
        image_index
        for image_index in range(split_start, split_end)
        if image_index not in indexed_paths
    ]
    if missing_indices:
        preview = ", ".join(f"{image_index:05d}" for image_index in missing_indices[:5])
        raise FileNotFoundError(
            f"Missing FFHQ files under {image_root}. "
            f"First missing indices: {preview}."
        )

    return [indexed_paths[image_index] for image_index in range(split_start, split_end)]


class FFHQDataset(Dataset[torch.Tensor]):
    """Reads FFHQ images and returns normalized tensors for one split."""

    def __init__(
        self,
        data_config: DataConfig,
        split: Split,
    ) -> None:
        """Initializes the FFHQ dataset.

        Args:
            data_config: Dataset configuration with root path and split bounds.
            split: Dataset split to load, either ``"train"`` or ``"val"``.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported FFHQ split '{split}'.")

        self.data_config = data_config
        self.split = split
        self.image_root = _resolve_image_root(data_config)
        split_start, split_end = _get_split_bounds(data_config, split)
        self.image_paths = _build_split_image_paths(
            self.image_root,
            split_start,
            split_end,
        )
        if split == "train":
            self.transform = build_train_transform(data_config)
        else:
            self.transform = build_val_transform(data_config)  # val and test use same transform

    def __len__(self) -> int:
        """Returns the number of samples in the split."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Loads one FFHQ image and applies the configured transform.

        Args:
            index: Position inside the chosen split.

        Returns:
            The transformed image tensor, typically normalized to ``[-1, 1]``.
        """
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def build_ffhq_datasets(data_config: DataConfig) -> tuple[FFHQDataset, FFHQDataset]:
    """Builds the standard train and validation FFHQ datasets."""
    train_dataset = FFHQDataset(data_config=data_config, split="train")
    val_dataset = FFHQDataset(data_config=data_config, split="val")
    return train_dataset, val_dataset


def build_ffhq_test_dataset(data_config: DataConfig) -> FFHQDataset | None:
    """Builds the test FFHQ dataset, or returns None if not configured."""
    if data_config.test_split_end <= data_config.test_split_start:
        return None
    return FFHQDataset(data_config=data_config, split="test")
