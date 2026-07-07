"""Data modules for FFHQ inpainting training."""

from torch.utils.data import Dataset

from .collate import safe_collate
from .ffhq import FFHQDataset, build_ffhq_datasets, build_ffhq_test_dataset
from NullFlow.utils.config import DataConfig


def build_datasets(data_config: DataConfig) -> tuple[Dataset, Dataset]:
    """Builds the configured train and validation datasets."""
    if data_config.dataset_name == "ffhq":
        return build_ffhq_datasets(data_config)
    raise ValueError(f"Unsupported dataset_name '{data_config.dataset_name}'.")


__all__ = [
    "FFHQDataset",
    "build_datasets",
    "build_ffhq_datasets",
    "build_ffhq_test_dataset",
    "safe_collate",
]
