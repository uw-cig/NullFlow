"""Core PMF flow model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PMFFlowModel(nn.Module, ABC):
    """Defines the unconditional reconstruction-model interface."""

    def __init__(self) -> None:
        """Initializes the flow model interface."""
        super().__init__()

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor | float | int,
        t: torch.Tensor | float | int,
        *,
        return_velocity_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predicts a reconstruction estimate or PMF velocity pair.

        Args:
            z: Current image state with shape ``(B, C, H, W)``.
            r: Earlier time value for each batch element or a broadcastable scalar.
            t: Current time value for each batch element or a broadcastable scalar.
            return_velocity_pair: Whether to return the PMF velocity pair instead
                of the reconstruction estimate.

        Returns:
            A reconstruction estimate with the same shape as ``z`` or a PMF
            velocity pair.

        Raises:
            ValueError: If the input shapes or time constraints are invalid.
        """
        self._validate_state_tensor(z)

        batch_size = z.shape[0]
        r_tensor = self._normalize_time_input(
            name="r",
            value=r,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )
        t_tensor = self._normalize_time_input(
            name="t",
            value=t,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )

        if return_velocity_pair:
            self._validate_velocity_time_pair(r_tensor, t_tensor)
            velocity_fn = getattr(self, "predict_velocity_pair", None)
            if not callable(velocity_fn):
                raise ValueError(
                    "return_velocity_pair=True requires predict_velocity_pair(...)."
                )
            return velocity_fn(z, r_tensor, t_tensor)

        self._validate_time_pair(r_tensor, t_tensor)
        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        h_tensor = t_tensor - r_tensor

        reconstruction = self.predict_reconstruction(
            z=z,
            h=h_tensor,
            r=r_tensor,
            t=t_tensor,
        )
        if reconstruction.shape != z.shape:
            raise ValueError(
                "predict_reconstruction must return a tensor with the same shape as z."
            )
        return reconstruction

    @abstractmethod
    def predict_reconstruction(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts a reconstruction estimate from normalized model inputs."""

    @staticmethod
    def reshape_time_like_image(time_values: torch.Tensor) -> torch.Tensor:
        """Reshapes ``(B,)`` time values to image-broadcast form."""
        return time_values.reshape(-1, 1, 1, 1)

    @staticmethod
    def _validate_state_tensor(z: torch.Tensor) -> None:
        """Checks the input state tensor passed to the model."""
        if z.ndim != 4:
            raise ValueError("z must have shape (B, C, H, W).")
        if z.shape[0] <= 0:
            raise ValueError("z must contain at least one batch element.")
        if not torch.is_floating_point(z):
            raise ValueError("z must be a floating-point tensor.")

    @staticmethod
    def _normalize_time_input(
        name: str,
        value: torch.Tensor | float | int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalizes a scalar or tensor time input to shape ``(B,)``."""
        if torch.is_tensor(value):
            time_tensor = value.to(device=device, dtype=dtype)
        else:
            time_tensor = torch.tensor(value, device=device, dtype=dtype)

        if time_tensor.ndim == 0:
            return time_tensor.expand(batch_size)
        if time_tensor.numel() == 1:
            return time_tensor.reshape(1).expand(batch_size)
        if time_tensor.numel() == batch_size:
            return time_tensor.reshape(batch_size)

        raise ValueError(
            f"{name} must be a scalar or contain exactly one value per batch element."
        )

    @staticmethod
    def _validate_time_pair(r: torch.Tensor, t: torch.Tensor) -> None:
        """Checks the required ordering constraints for ``r`` and ``t``."""
        if torch.any(r < 0):
            raise ValueError("r must satisfy r >= 0.")
        if torch.any(t <= 0):
            raise ValueError("t must satisfy t > 0.")
        if torch.any(t > 1):
            raise ValueError("t must satisfy t <= 1.")
        if torch.any(r >= t):
            raise ValueError("r and t must satisfy r < t.")

    @staticmethod
    def _validate_velocity_time_pair(r: torch.Tensor, t: torch.Tensor) -> None:
        """Checks the ordering constraints for PMF velocity queries."""
        if torch.any(r < 0):
            raise ValueError("r must satisfy r >= 0.")
        if torch.any(t <= 0):
            raise ValueError("t must satisfy t > 0.")
        if torch.any(t > 1):
            raise ValueError("t must satisfy t <= 1.")
        if torch.any(r > t):
            raise ValueError("velocity queries must satisfy r <= t.")
