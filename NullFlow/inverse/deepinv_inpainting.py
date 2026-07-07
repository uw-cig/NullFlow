"""deepinv-based inpainting helpers."""

from __future__ import annotations

import deepinv as dinv
import torch

from NullFlow.inverse.masks import (
    FixedMaskSource,
    InpaintingMaskGenerator,
    build_mask_generator,
)
from NullFlow.utils.config import DataConfig


class DeepInvInpainting:
    """Wraps deepinv inpainting physics for the current project."""

    def __init__(
        self,
        data_config: DataConfig,
        *,
        mask_generator: InpaintingMaskGenerator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initializes the inpainting physics wrapper.

        Args:
            data_config: Data configuration with image shape information.
            mask_generator: Optional mask generator override.
            device: Device used by the deepinv physics operator.
            dtype: Default dtype for sampled mask tensors.
        """
        self.data_config = data_config
        self.mask_generator = mask_generator or build_mask_generator(data_config)
        self.device = torch.device(device)
        self.dtype = dtype
        self.physics = dinv.physics.Inpainting(
            img_size=(
                data_config.image_channels,
                data_config.image_size,
                data_config.image_size,
            ),
            mask=None,
            device=self.device,
        )

    def _build_params(
        self,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Builds the parameter payload used by later inpainting calls."""
        if not self.data_config.precompute_a_dagger_ops:
            return params

        mask = params["mask"]
        return {
            **params,
            "a_dagger": mask.clone(),
            "a_dagger_a": mask.clone(),
        }

    def sample_params(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        smps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Builds a batch of inpainting parameters for training."""
        del smps
        if mask is not None:
            params = {
                "mask": mask.to(
                    device=self.device if device is None else device,
                    dtype=self.dtype if dtype is None else dtype,
                )
            }
        else:
            params = self.mask_generator.sample(
                batch_size=batch_size,
                device=self.device if device is None else device,
                dtype=self.dtype if dtype is None else dtype,
            )
        return self._build_params(params)

    def sample_validation_params(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        smps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Builds a deterministic batch of inpainting parameters for validation."""
        del smps
        if mask is not None:
            params = {
                "mask": mask.to(
                    device=self.device if device is None else device,
                    dtype=self.dtype if dtype is None else dtype,
                )
            }
        else:
            params = self.mask_generator.sample_validation(
                batch_size=batch_size,
                device=self.device if device is None else device,
                dtype=self.dtype if dtype is None else dtype,
            )
        return self._build_params(params)

    def measurement_for_visualization(
        self,
        y: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns the visualization tensor used for the measurement row."""
        del params
        return y

    def resolve_mask_metadata(
        self,
        params: dict[str, torch.Tensor],
    ) -> dict[str, tuple[str, ...]]:
        """Returns fixed-mask metadata for one sampled parameter batch."""
        source = getattr(self.mask_generator, "source", None)
        mask_index = params.get("mask_index")
        if not isinstance(source, FixedMaskSource) or mask_index is None:
            return {}
        return source.describe(mask_index)

    def blend_reconstruction(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Keeps the model prediction only inside the missing region."""
        observed_mask = params["mask"]
        missing_mask = 1.0 - observed_mask
        return observed_mask * ground_truth + missing_mask * prediction

    def project_for_loss(
        self,
        tensor: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Projects one tensor to the missing image region used by inpainting loss."""
        return (1.0 - params["mask"]) * tensor

    def update(self, params: dict[str, torch.Tensor]) -> None:
        """Updates the deepinv physics object with the current mask."""
        self.physics.update(mask=params["mask"])

    def measure(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Applies the forward inpainting operator."""
        self.update(params)
        return self.physics.A(x)

    def pseudo_inverse(
        self,
        y: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Applies the inpainting pseudo-inverse operator."""
        if "a_dagger" in params:
            return params["a_dagger"] * y
        self.update(params)
        return self.physics.A_dagger(y)

    def nullspace_component(
        self,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the component of ``e`` in the nullspace of ``A``."""
        if "a_dagger_a" in params:
            observed_component = params["a_dagger_a"] * e
            return e - observed_component
        self.update(params)
        observed_component = self.physics.A_dagger(self.physics.A(e))
        return e - observed_component

    def null_initializer(
        self,
        y: torch.Tensor,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Builds ``A^dagger y + (I - A^dagger A)e`` for training or inference."""
        observed_signal = self.pseudo_inverse(y, params)
        missing_signal = self.nullspace_component(e, params)
        return observed_signal + missing_signal


def build_deepinv_inpainting(
    data_config: DataConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DeepInvInpainting:
    """Builds the default deepinv inpainting wrapper."""
    return DeepInvInpainting(
        data_config=data_config,
        device=device,
        dtype=dtype,
    )
