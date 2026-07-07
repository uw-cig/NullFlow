"""Utilities for reproducible random number generation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Sets Python, NumPy, and PyTorch random seeds.

    Args:
        seed: Global random seed. Must be non-negative.
        deterministic: Whether to enable deterministic PyTorch behavior.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Sets NumPy and Python seeds inside a DataLoader worker.

    Args:
        worker_id: DataLoader worker id. The value is unused directly because
            PyTorch already folds it into ``torch.initial_seed()``.
    """
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Builds a CPU generator seeded for DataLoader or sampling.

    Args:
        seed: Random seed used to initialize the generator.

    Returns:
        A seeded ``torch.Generator`` instance.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


__all__ = [
    "make_torch_generator",
    "seed_worker",
    "set_seed",
]
