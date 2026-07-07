"""Transform builders for FFHQ image preprocessing."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from NullFlow.utils.config import DataConfig


def _resolve_interpolation_mode(interpolation_name: str) -> InterpolationMode:
    """Maps a config string to a torchvision interpolation mode.

    Args:
        interpolation_name: String identifier from :class:`DataConfig`.

    Returns:
        The matching torchvision interpolation mode.

    Raises:
        ValueError: If the interpolation mode is not supported.
    """
    interpolation_map = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    normalized_name = interpolation_name.strip().lower()
    try:
        return interpolation_map[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(interpolation_map))
        raise ValueError(
            f"Unsupported resize_interpolation '{interpolation_name}'. "
            f"Expected one of: {supported}."
        ) from error


def _make_range_mapper(value_range: tuple[float, float]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Builds a tensor mapper from ``[0, 1]`` to the configured range.

    Args:
        value_range: Target interval for image tensors after ``ToTensor``.

    Returns:
        A callable that maps a float tensor from ``[0, 1]`` to ``value_range``.
    """
    value_min, value_max = value_range
    scale = value_max - value_min

    def map_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Applies an affine remapping to the input tensor."""
        return tensor * scale + value_min

    return map_tensor


def _build_resize_transform(data_config: DataConfig) -> transforms.Resize:
    """Creates the shared FFHQ resize transform."""
    interpolation = _resolve_interpolation_mode(data_config.resize_interpolation)
    return transforms.Resize(
        size=(data_config.image_size, data_config.image_size),
        interpolation=interpolation,
    )


def build_train_transform(data_config: DataConfig) -> transforms.Compose:
    """Builds the FFHQ training transform pipeline.

    The pipeline resizes each image to the target square resolution without cropping,
    optionally flips it horizontally, converts it to a tensor, and maps
    pixel intensities to ``normalization_range``.

    Args:
        data_config: Typed data configuration for the current experiment.

    Returns:
        A torchvision ``Compose`` transform for training samples.
    """
    transform_steps: list[Callable[..., object]] = [_build_resize_transform(data_config)]
    if data_config.enable_random_flip:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(_make_range_mapper(data_config.normalization_range)),
        ]
    )
    return transforms.Compose(transform_steps)


def build_val_transform(data_config: DataConfig) -> transforms.Compose:
    """Builds the FFHQ validation transform pipeline.

    Args:
        data_config: Typed data configuration for the current experiment.

    Returns:
        A torchvision ``Compose`` transform for validation samples.
    """
    return transforms.Compose(
        [
            _build_resize_transform(data_config),
            transforms.ToTensor(),
            transforms.Lambda(_make_range_mapper(data_config.normalization_range)),
        ]
    )
