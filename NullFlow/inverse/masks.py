"""Mask loading and sampling helpers for inpainting and MRI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from NullFlow.utils.config import DataConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTER_MASK_SIZE = 64
_VALID_MASK_SUFFIXES = {".pt", ".pth", ".npy", ".npz"}


def _resolve_repo_path(path_value: str | Path) -> Path:
    """Resolves one configured path relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _get_center_bounds(image_size: int, mask_size: int) -> tuple[int, int]:
    """Computes the inclusive-exclusive bounds for a centered square hole."""
    if mask_size <= 0:
        raise ValueError("mask_size must be positive.")
    if mask_size > image_size:
        raise ValueError("mask_size must not exceed image_size.")

    start = (image_size - mask_size) // 2
    end = start + mask_size
    if start < 0 or end > image_size:
        raise ValueError("The centered mask must lie inside the image.")
    return start, end


def _load_mask_payload(
    path_value: str | Path,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Loads one fixed-mask payload from disk and returns it as a tensor."""
    path = _resolve_repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {path}")

    if path.suffix == ".npy":
        payload = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        if key is None:
            raise ValueError("npz mask files require fixed_mask_key.")
        if key not in payload:
            raise KeyError(f"Mask key '{key}' not found in {path}.")
        payload = payload[key]
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if key is not None:
            if not isinstance(payload, dict) or key not in payload:
                raise KeyError(f"Mask key '{key}' not found in {path}.")
            payload = payload[key]

    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu()
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(np.asarray(payload))
    raise ValueError("Mask files must store a tensor or a NumPy array payload.")


def _normalize_image_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    image_channels: int,
) -> torch.Tensor:
    """Normalizes one fixed image-space mask to shape (channels, height, width)."""
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            "Image-space masks must have shape (height, width), (1, height, width), or (channels, height, width)."
        )
    if mask.shape[-2:] != (image_size, image_size):
        raise ValueError(
            f"Image-space masks must have spatial shape {(image_size, image_size)}."
        )
    if mask.shape[0] == 1:
        mask = mask.expand(image_channels, -1, -1)
    elif mask.shape[0] != image_channels:
        raise ValueError(
            f"Image-space masks must have 1 or {image_channels} channels, but found {mask.shape[0]}."
        )
    return mask.to(dtype=torch.float32).contiguous()


def normalize_kspace_mask(
    mask: torch.Tensor,
    *,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    """Normalizes one fixed k-space mask to shape (1, height, width)."""
    if isinstance(image_size, int):
        height = width = image_size
    else:
        if len(image_size) != 2:
            raise ValueError("image_size tuples must contain exactly two dimensions.")
        height, width = int(image_size[0]), int(image_size[1])
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "K-space masks must have shape (height, width), (1, height, width), or (1, 1, height, width)."
        )
    if mask.shape[-2:] != (height, width):
        raise ValueError(f"K-space masks must have spatial shape {(height, width)}.")
    return mask.to(dtype=torch.float32).contiguous()


def _discover_mask_paths(mask_bank_dir: str | Path) -> tuple[Path, ...]:
    """Discovers fixed-mask files from one bank directory."""
    root = _resolve_repo_path(mask_bank_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Mask bank directory does not exist: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in _VALID_MASK_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"Mask bank directory does not contain any supported mask files: {root}")
    return paths


@dataclass(frozen=True)
class FixedMaskSource:
    """Stores one fixed mask or fixed mask bank in normalized tensor form."""

    masks: tuple[torch.Tensor, ...]
    ids: tuple[str, ...]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the stored masks."""
        if not self.masks:
            raise ValueError("FixedMaskSource must contain at least one mask.")
        if len(self.masks) != len(self.ids):
            raise ValueError("FixedMaskSource ids must match the number of masks.")
        if len(self.masks) != len(self.sources):
            raise ValueError("FixedMaskSource sources must match the number of masks.")
        first_shape = tuple(self.masks[0].shape)
        for mask in self.masks:
            if tuple(mask.shape) != first_shape:
                raise ValueError("All masks in one FixedMaskSource must share the same shape.")

    def sample(
        self,
        batch_size: int,
        *,
        deterministic: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples one batch of masks and returns the mask indices."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if deterministic:
            indices = torch.arange(batch_size, dtype=torch.long) % len(self.masks)
        else:
            indices = torch.randint(len(self.masks), (batch_size,), dtype=torch.long)
        batch = torch.stack([self.masks[int(index)] for index in indices.tolist()], dim=0)
        return batch.to(device=device, dtype=dtype), indices

    def describe(
        self,
        indices: torch.Tensor,
    ) -> dict[str, tuple[str, ...]]:
        """Returns the ids and source paths for the sampled masks."""
        flat_indices = indices.detach().cpu().reshape(-1).tolist()
        return {
            "mask_ids": tuple(self.ids[int(index)] for index in flat_indices),
            "mask_sources": tuple(self.sources[int(index)] for index in flat_indices),
        }


def _build_fixed_mask_source(
    *,
    fixed_mask_path: str | None,
    fixed_mask_bank_dir: str | None,
    fixed_mask_key: str | None,
    normalize_fn,
) -> FixedMaskSource:
    """Builds one normalized fixed-mask source from a single file or a bank directory."""
    if fixed_mask_path is not None:
        normalized = normalize_fn(_load_mask_payload(fixed_mask_path, key=fixed_mask_key))
        resolved = _resolve_repo_path(fixed_mask_path)
        return FixedMaskSource(
            masks=(normalized,),
            ids=(resolved.stem,),
            sources=(str(resolved),),
        )

    if fixed_mask_bank_dir is not None:
        paths = _discover_mask_paths(fixed_mask_bank_dir)
        masks = tuple(
            normalize_fn(_load_mask_payload(path, key=fixed_mask_key))
            for path in paths
        )
        ids = tuple(path.stem for path in paths)
        sources = tuple(str(path.resolve()) for path in paths)
        return FixedMaskSource(masks=masks, ids=ids, sources=sources)

    raise ValueError("A fixed mask path or bank directory must be provided.")


def build_fixed_image_mask_source(data_config: DataConfig) -> FixedMaskSource:
    """Builds one fixed image-space mask source from config."""
    return _build_fixed_mask_source(
        fixed_mask_path=data_config.fixed_mask_path,
        fixed_mask_bank_dir=data_config.fixed_mask_bank_dir,
        fixed_mask_key=data_config.fixed_mask_key,
        normalize_fn=lambda mask: _normalize_image_mask(
            mask,
            image_size=data_config.image_size,
            image_channels=data_config.image_channels,
        ),
    )


def build_fixed_kspace_mask_source(
    *,
    image_size: int | tuple[int, int],
    fixed_mask_path: str | None = None,
    fixed_mask_bank_dir: str | None = None,
    fixed_mask_key: str | None = None,
    fixed_s_path: str | None = None,
) -> FixedMaskSource:
    """Builds one fixed k-space mask source from MRI config values."""
    return _build_fixed_mask_source(
        fixed_mask_path=fixed_mask_path or fixed_s_path,
        fixed_mask_bank_dir=fixed_mask_bank_dir,
        fixed_mask_key=fixed_mask_key,
        normalize_fn=lambda mask: normalize_kspace_mask(mask, image_size=image_size),
    )


def build_center_square_mask(
    batch_size: int,
    image_size: int,
    image_channels: int,
    mask_size: int = DEFAULT_CENTER_MASK_SIZE,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Builds an inpainting mask with a missing center square."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if image_channels <= 0:
        raise ValueError("image_channels must be positive.")

    start, end = _get_center_bounds(image_size, mask_size)
    mask = torch.ones(
        (batch_size, image_channels, image_size, image_size),
        device=device,
        dtype=dtype,
    )
    mask[:, :, start:end, start:end] = 0
    return mask


class InpaintingMaskGenerator(Protocol):
    """Protocol implemented by inpainting mask generators."""

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds one batch of training mask parameters."""

    def sample_validation(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds one deterministic batch of validation mask parameters."""


@dataclass(frozen=True)
class CenterSquareMaskGenerator:
    """Builds the fixed center-square inpainting mask used in V1."""

    image_size: int
    image_channels: int
    mask_size: int = DEFAULT_CENTER_MASK_SIZE

    def __post_init__(self) -> None:
        """Validates the center-square mask configuration."""
        _get_center_bounds(self.image_size, self.mask_size)
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive.")

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds a batch of training mask parameters."""
        return {
            "mask": build_center_square_mask(
                batch_size=batch_size,
                image_size=self.image_size,
                image_channels=self.image_channels,
                mask_size=self.mask_size,
                device=device,
                dtype=dtype,
            )
        }

    def sample_validation(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds the fixed validation mask batch."""
        return self.sample(batch_size=batch_size, device=device, dtype=dtype)


@dataclass(frozen=True)
class FixedMaskGenerator:
    """Builds inpainting masks from one fixed mask file or bank."""

    source: FixedMaskSource

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds one training batch from the fixed mask source."""
        mask, indices = self.source.sample(
            batch_size,
            deterministic=False,
            device=device,
            dtype=dtype,
        )
        return {
            "mask": mask,
            "mask_index": indices.to(device=device),
        }

    def sample_validation(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Builds one deterministic validation batch from the fixed mask source."""
        mask, indices = self.source.sample(
            batch_size,
            deterministic=True,
            device=device,
            dtype=dtype,
        )
        return {
            "mask": mask,
            "mask_index": indices.to(device=device),
        }


def build_mask_generator(data_config: DataConfig) -> InpaintingMaskGenerator:
    """Builds the configured inpainting mask generator from data settings."""
    if data_config.mask_type == "center":
        return CenterSquareMaskGenerator(
            image_size=data_config.image_size,
            image_channels=data_config.image_channels,
            mask_size=data_config.center_mask_size,
        )
    if data_config.mask_type == "fixed":
        return FixedMaskGenerator(build_fixed_image_mask_source(data_config))
    raise ValueError(f"Unsupported mask_type '{data_config.mask_type}'.")
