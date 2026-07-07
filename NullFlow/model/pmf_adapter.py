"""Adapter and factory helpers for the UNet PMF backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from NullFlow.model.pmf_flow import PMFFlowModel
from NullFlow.model.unet_pmf import ConditionedUNetPMFVelocityPairModel
from NullFlow.utils.config import DataConfig, ModelConfig


MIN_TIME_SCALE = 0.05


@dataclass
class PMFConditioningState:
    """Stores optional conditioning inputs for the upstream PMF model."""

    omega: torch.Tensor | float | int | None = None
    t_min: torch.Tensor | float | int | None = None
    t_max: torch.Tensor | float | int | None = None
    labels: torch.Tensor | int | None = None


def _resolve_unet_emb_channels(model_config: ModelConfig) -> int:
    """Returns the guided-diffusion-style embedding width for the UNet backend."""
    if model_config.unet_emb_channels is not None:
        return model_config.unet_emb_channels
    return model_config.unet_model_channels * 4


def _build_unet_pmf_model(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> ConditionedUNetPMFVelocityPairModel:
    """Builds the local UNet-based PMF velocity-pair model."""
    use_label_conditioning = model_config.conditional and model_config.unet_use_label_conditioning
    return ConditionedUNetPMFVelocityPairModel(
        image_size=data_config.image_size,
        in_channels=data_config.image_channels,
        emb_channels=_resolve_unet_emb_channels(model_config),
        model_channels=model_config.unet_model_channels,
        num_res_blocks=model_config.unet_num_res_blocks,
        attention_resolutions=model_config.unet_attention_resolutions,
        dropout=model_config.unet_dropout,
        channel_mult=model_config.unet_channel_mult,
        conv_resample=model_config.unet_conv_resample,
        use_checkpoint=model_config.unet_use_checkpoint,
        use_fp16=model_config.unet_use_fp16,
        num_heads=model_config.unet_num_heads,
        num_head_channels=model_config.unet_num_head_channels,
        num_heads_upsample=model_config.unet_num_heads_upsample,
        use_scale_shift_norm=model_config.unet_use_scale_shift_norm,
        resblock_updown=model_config.unet_resblock_updown,
        use_new_attention_order=model_config.unet_use_new_attention_order,
        num_classes=model_config.num_classes if use_label_conditioning else None,
        use_aux_conditioning=model_config.unet_use_aux_conditioning,
        use_label_conditioning=use_label_conditioning,
    )


def _validate_model_spec(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> None:
    """Checks the project-side assumptions for the selected PMF preset."""
    model_name = model_config.model_name
    if model_name != "pmf_unet":
        raise ValueError(f"Unsupported PMF model preset '{model_name}'.")
    if data_config.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if data_config.image_channels <= 0:
        raise ValueError("image_channels must be positive.")


class PMFAdapter(PMFFlowModel):
    """Wraps the configured PMF backbone behind the local flow interface."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        *,
        eval_mode: bool = False,
    ) -> None:
        """Initializes the PMF adapter.

        Args:
            model_config: Model selection and conditioning configuration.
            data_config: Data configuration with image shape information.
            eval_mode: Whether to instantiate the upstream model in eval mode.
        """
        super().__init__()
        _validate_model_spec(model_config, data_config)

        self.model_name = model_config.model_name
        self.model_config = model_config
        self.data_config = data_config
        self.conditioning = PMFConditioningState()
        self.net = _build_unet_pmf_model(model_config, data_config)
        if eval_mode and hasattr(self.net, "eval"):
            self.net.eval()
        elif hasattr(self.net, "train"):
            self.net.train()

    def set_conditioning(
        self,
        *,
        omega: torch.Tensor | float | int | None = None,
        t_min: torch.Tensor | float | int | None = None,
        t_max: torch.Tensor | float | int | None = None,
        labels: torch.Tensor | int | None = None,
    ) -> None:
        """Stores optional conditioning values for future forward passes."""
        self.conditioning = PMFConditioningState(
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )

    def clear_conditioning(self) -> None:
        """Clears any previously stored PMF conditioning values."""
        self.conditioning = PMFConditioningState()

    def predict_reconstruction(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Converts the PMF average velocity back to a reconstruction estimate."""
        del r
        u, _ = self._predict_velocities(
            z=z,
            t=t,
            h=h,
        )
        time_scale = torch.clamp(self.reshape_time_like_image(t), min=MIN_TIME_SCALE)
        return z - time_scale * u

    def predict_velocity_pair(
        self,
        z: torch.Tensor,
        r: torch.Tensor | float | int,
        t: torch.Tensor | float | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the upstream PMF velocity pair for one batch."""
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
        self._validate_velocity_times(r_tensor, t_tensor)
        h_tensor = t_tensor - r_tensor

        return self._predict_velocities(
            z=z,
            t=t_tensor,
            h=h_tensor,
        )

    def predict_imf_velocity(
        self,
        z: torch.Tensor,
        r: torch.Tensor | float | int,
        t: torch.Tensor | float | int,
    ) -> torch.Tensor:
        """Returns the single velocity used by the IMF objective.

        IMF evaluates ``fn(z, t, t)`` for the instantaneous velocity seed. Keep
        that zero-h query local to this explicit helper instead of relaxing the
        reconstruction ``forward(...)`` contract, which still requires ``r < t``.
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
        self._validate_velocity_times(r_tensor, t_tensor)
        h_tensor = t_tensor - r_tensor

        u, _ = self._predict_velocities(
            z=z,
            t=t_tensor,
            h=h_tensor,
        )
        return u

    def predict_head_pair(
        self,
        z: torch.Tensor,
        r: torch.Tensor | float | int,
        t: torch.Tensor | float | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the PMF velocity pair through the legacy accessor name."""
        return self.predict_velocity_pair(z, r, t)

    @staticmethod
    def _validate_velocity_times(r: torch.Tensor, t: torch.Tensor) -> None:
        """Checks the time constraints for PMF velocity queries."""
        if torch.any(r < 0):
            raise ValueError("r must satisfy r >= 0.")
        if torch.any(t <= 0):
            raise ValueError("t must satisfy t > 0.")
        if torch.any(t > 1):
            raise ValueError("t must satisfy t <= 1.")
        if torch.any(r > t):
            raise ValueError("predict_velocity_pair requires r <= t.")

    def _predict_velocities(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the upstream PMF velocity pair for one batch."""
        batch_size = z.shape[0]
        omega, t_min, t_max, labels = self._resolve_conditioning(
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )

        u, v = self.net(
            z,
            t=t.reshape(batch_size),
            h=h.reshape(batch_size),
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        return u, v

    def _resolve_conditioning(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the upstream PMF conditioning tensors for one batch."""
        omega = self._normalize_optional_input(
            name="omega",
            value=self.conditioning.omega,
            default=1.0,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        t_min = self._normalize_optional_input(
            name="t_min",
            value=self.conditioning.t_min,
            default=0.0,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        t_max = self._normalize_optional_input(
            name="t_max",
            value=self.conditioning.t_max,
            default=1.0,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        labels = self._normalize_optional_input(
            name="labels",
            value=self.conditioning.labels,
            default=0,
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        self._validate_conditioning(omega=omega, t_min=t_min, t_max=t_max, labels=labels)
        return omega, t_min, t_max, labels

    @staticmethod
    def _normalize_optional_input(
        name: str,
        value: torch.Tensor | float | int | None,
        default: torch.Tensor | float | int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalizes an optional scalar or batch input to shape ``(B,)``."""
        normalized_value = default if value is None else value
        if torch.is_tensor(normalized_value):
            tensor = normalized_value.to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(normalized_value, device=device, dtype=dtype)

        if tensor.ndim == 0:
            return tensor.expand(batch_size)
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(batch_size)
        if tensor.numel() == batch_size:
            return tensor.reshape(batch_size)

        raise ValueError(
            f"{name} must be a scalar or contain exactly one value per batch element."
        )

    @staticmethod
    def _validate_conditioning(
        *,
        omega: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Checks the PMF conditioning tensors before dispatching upstream."""
        if torch.any(omega <= 0):
            raise ValueError("omega must be positive.")
        if torch.any(t_min < 0):
            raise ValueError("t_min must satisfy t_min >= 0.")
        if torch.any(t_max > 1):
            raise ValueError("t_max must satisfy t_max <= 1.")
        if torch.any(t_min > t_max):
            raise ValueError("t_min must satisfy t_min <= t_max.")
        if torch.any(labels < 0):
            raise ValueError("labels must be non-negative.")


def build_pmf_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    *,
    eval_mode: bool = False,
) -> PMFAdapter:
    """Builds the configured PMF adapter model."""
    return PMFAdapter(
        model_config=model_config,
        data_config=data_config,
        eval_mode=eval_mode,
    )
