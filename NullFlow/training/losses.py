"""Training loss helpers for one-step PMF inpainting."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from NullFlow.model.pmf_flow import PMFFlowModel
from NullFlow.training.time_sampling import sample_t_r

if TYPE_CHECKING:
    from NullFlow.inverse.deepinv_inpainting import DeepInvInpainting


def _reshape_like_image(time_values: torch.Tensor) -> torch.Tensor:
    """Reshapes ``(B,)`` time values for image broadcasting."""
    return PMFFlowModel.reshape_time_like_image(time_values)


def _reshape_like_kspace(time_values: torch.Tensor) -> torch.Tensor:
    """Reshapes ``(B,)`` time values for k-space broadcasting (B, 1, 1, 1, 1)."""
    return time_values[:, None, None, None, None]


def _to_unit_interval(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Maps normalized images to ``[0, 1]``."""
    value_min, value_max = value_range
    scale = value_max - value_min
    if scale <= 0:
        raise ValueError("value_range must be an increasing interval.")
    return ((images - value_min) / scale).clamp(0.0, 1.0)


def _missing_region_mask(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Builds the broadcast mask for the missing inpainting region."""
    if "mask" not in params:
        raise ValueError("params must contain a mask tensor.")
    return 1.0 - params["mask"]


def _disable_cuda_autocast(device: torch.device):
    """Disables CUDA autocast inside numerically sensitive PMF dynamics code."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _get_velocities(model: PMFFlowModel):
    """Returns the PMF velocity-pair accessor required by the training loss."""
    direct_velocity_fn = getattr(model, "predict_velocity_pair", None)

    def velocity_pair_fn(
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if callable(direct_velocity_fn):
            return direct_velocity_fn(z, r, t)
        if callable(model):
            outputs = model(z, r, t, return_velocity_pair=True)
        else:
            raise ValueError(
                "OneStepLoss requires a callable model or a predict_velocity_pair(...) method."
            )
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError(
                "return_velocity_pair=True must return a tuple of (u, v)."
            )
        return outputs

    return velocity_pair_fn


def _get_imf_velocity(model: PMFFlowModel):
    """Returns the single-velocity IMF accessor.

    Existing PMF backbones expose a velocity pair. IMF uses the first head as
    ``fn(z, r, t)`` so the training objective can be added without changing the
    model architecture yet.
    """
    direct_imf_velocity_fn = getattr(model, "predict_imf_velocity", None)
    if callable(direct_imf_velocity_fn) and not hasattr(model, "module"):
        return direct_imf_velocity_fn

    velocity_pair_fn = _get_velocities(model)

    def velocity_fn(
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        u, _ = velocity_pair_fn(z, r, t)
        return u

    return velocity_fn


class _LPIPSLoss:
    """Wraps LPIPS so gradients flow only through the predicted image."""

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self._model = model
        self.value_range = value_range

    def _get_model(self, device: torch.device) -> nn.Module:
        """Returns the frozen LPIPS network on the requested device."""
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net="alex")
        self._model = self._model.to(device=device)
        self._model.eval()
        self._model.requires_grad_(False)
        return self._model

    def _to_lpips_input(self, x: torch.Tensor) -> torch.Tensor:
        """Converts any image tensor to 3-channel [0, 1] for LPIPS.

        For 3-channel inputs, rescales from value_range to [0, 1].
        For other channel counts (e.g. 2-channel complex MRI), computes the
        per-pixel L2 magnitude, normalizes to [0, 1], and repeats to 3 channels.
        """
        lo, hi = self.value_range
        if x.shape[1] == 3:
            return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        mag = x.norm(dim=1, keepdim=True)
        max_mag = max(abs(lo), abs(hi)) * (x.shape[1] ** 0.5)
        return (mag / max_mag).clamp(0.0, 1.0).repeat(1, 3, 1, 1)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes one LPIPS score per image."""
        model = self._get_model(prediction.device)
        pred_in = self._to_lpips_input(prediction)
        with torch.no_grad():
            tgt_in = self._to_lpips_input(target)
        scores = model(pred_in, tgt_in)
        return scores.reshape(-1)


class _ConvNeXtFeatureExtractor(nn.Module):
    """Returns pooled ConvNeXt features for perceptual comparisons."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns one pooled feature vector per image."""
        features = self.backbone.features(images)
        pooled = self.backbone.avgpool(features)
        return pooled.reshape(pooled.shape[0], -1)


class _ConvNeXtLoss:
    """Computes a frozen ConvNeXt feature distance."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self._model = model
        self.value_range = value_range

    def _get_model(self, device: torch.device) -> nn.Module:
        """Returns the frozen ConvNeXt feature extractor on the requested device."""
        if self._model is None:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self._model = _ConvNeXtFeatureExtractor(backbone)
        self._model = self._model.to(device=device)
        self._model.eval()
        self._model.requires_grad_(False)
        return self._model

    def _to_3ch_unit(self, images: torch.Tensor) -> torch.Tensor:
        """Converts any image to 3-channel [0, 1].

        For 3-channel inputs, rescales from value_range to [0, 1].
        For other channel counts (e.g. 2-channel complex MRI), computes the
        per-pixel L2 magnitude, normalizes to [0, 1], and repeats to 3 channels.
        """
        lo, hi = self.value_range
        if images.shape[1] == 3:
            return _to_unit_interval(images, self.value_range)
        mag = images.norm(dim=1, keepdim=True)
        max_mag = max(abs(lo), abs(hi)) * (images.shape[1] ** 0.5)
        return (mag / max_mag).clamp(0.0, 1.0).repeat(1, 3, 1, 1)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the pretrained ConvNeXt input normalization."""
        unit_images = self._to_3ch_unit(images)
        mean = torch.tensor(
            self._MEAN,
            device=images.device,
            dtype=images.dtype,
        ).reshape(1, -1, 1, 1)
        std = torch.tensor(
            self._STD,
            device=images.device,
            dtype=images.dtype,
        ).reshape(1, -1, 1, 1)
        return (unit_images - mean) / std

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes one ConvNeXt feature-space MSE per image."""
        model = self._get_model(prediction.device)
        prediction_input = self._normalize(prediction)
        target_input = self._normalize(target)
        prediction_features = model(prediction_input)
        with torch.no_grad():
            target_features = model(target_input)
        return torch.mean((prediction_features - target_features) ** 2, dim=1)


class OneStepLoss:
    """Builds the one-step PMF inpainting loss for one model and physics pair."""

    def __init__(
        self,
        model: PMFFlowModel,
        physics: DeepInvInpainting,
        *,
        project_loss_to_measurement_space: bool = True,
        lpips: bool = False,
        lpips_lambda: float = 1.0,
        convnext: bool = False,
        convnext_lambda: float = 0.0,
        reconstruction_mse_lambda: float = 0.0,
        perceptual_max_t: float = 1.0,
        mri_noise_sigma: float = 1.0,
        value_range: tuple[float, float] = (-1.0, 1.0),
        lpips_model: nn.Module | None = None,
        convnext_model: nn.Module | None = None,
    ) -> None:
        """Initializes the reusable loss builder."""
        self.model = model
        self.physics = physics
        self._velocities = _get_velocities(model)
        self.project_loss_to_measurement_space = project_loss_to_measurement_space
        self.lpips = lpips
        self.lpips_lambda = lpips_lambda
        self.convnext = convnext
        self.convnext_lambda = convnext_lambda
        self.reconstruction_mse_lambda = reconstruction_mse_lambda
        self.perceptual_max_t = perceptual_max_t
        self.mri_noise_sigma = mri_noise_sigma
        self.value_range = value_range
        self._lpips_loss = _LPIPSLoss(model=lpips_model, value_range=value_range) if lpips else None
        self._convnext_loss = (
            _ConvNeXtLoss(model=convnext_model, value_range=value_range)
            if convnext
            else None
        )

    def _u_fn(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the PMF velocity pair used by the training dynamics."""
        return self._velocities(z, r, t)

    def _make_kspace_velocity_fn(
        self,
        params: dict[str, torch.Tensor],
    ):
        """Returns a velocity pair function operating entirely in multi-coil k-space.

        Implements the k-space u_fn from Algorithm 1:
            x       = C*F⁻¹ · z_k
            x_hat   = net(x, r, t)
            x_hat_k = F·C · x_hat
            u_k     = (z_k - x_hat_k) / t

        ``v_k`` is the same u_fn evaluated at ``(z_k, t, t)``.
        """
        state_to_model_input = getattr(self.physics, "state_to_model_input", None)
        model_output_to_state = getattr(self.physics, "model_output_to_state", None)
        if not callable(state_to_model_input) or not callable(model_output_to_state):
            raise ValueError(
                "MRI k-space loss requires physics.state_to_model_input(...) "
                "and physics.model_output_to_state(...)."
            )
        # Use the underlying module to avoid DDP's _DDPSink which is incompatible
        # with torch.func transforms (jvp). Safe here because this fn runs under no_grad.
        base_model = getattr(self.model, "module", self.model)

        def kspace_u(
            z_k: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            x_img = state_to_model_input(z_k, params)
            x_hat_img = base_model(x_img, r, t)
            x_hat_k = model_output_to_state(x_hat_img, params)
            return (z_k - x_hat_k) / _reshape_like_kspace(t)

        def kspace_vel_pair(
            z_k: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            u = kspace_u(z_k, r, t)
            v = u  # model requires r < t, so (t,t) is invalid; use current velocity as seed
            return u, v

        return kspace_vel_pair

    def _compute_detached_dudt(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        velocity_fn=None,
    ) -> torch.Tensor:
        """Computes ``dudt`` outside the main autograd graph to save memory.

        Args:
            velocity_fn: Optional replacement for the default ``self._u_fn``.
                Pass the k-space velocity function for the MRI k-space path.
        """
        vfn = self._u_fn if velocity_fn is None else velocity_fn
        with torch.no_grad():
            z_detached = z.detach()
            r_detached = r.detach()
            t_detached = t.detach()
            v_seed = vfn(z_detached, r_detached, t_detached)[1].detach()
            (_, _), (dudt, _) = torch.func.jvp(
                vfn,
                (z_detached, r_detached, t_detached),
                (
                    v_seed,
                    torch.zeros_like(r_detached),
                    torch.ones_like(t_detached),
                ),
            )
        return dudt

    def _extract_batch_inputs(
        self,
        batch: torch.Tensor | Mapping[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Returns the image tensor plus any physics-side auxiliary inputs."""
        if torch.is_tensor(batch):
            return batch, {}, None
        x = batch.get("x")
        if not torch.is_tensor(x):
            raise ValueError("mapping batches must contain a tensor 'x'.")
        extra_inputs: dict[str, torch.Tensor] = {}
        smps = batch.get("smps")
        if torch.is_tensor(smps):
            extra_inputs["smps"] = smps
        mask = batch.get("mask")
        if torch.is_tensor(mask):
            extra_inputs["mask"] = mask
        y_hat = batch.get("y_hat")
        if y_hat is not None and not torch.is_tensor(y_hat):
            raise ValueError("mapping batches must use a tensor 'y_hat' when provided.")
        return x, extra_inputs, y_hat

    def __call__(
        self,
        batch: torch.Tensor | Mapping[str, object],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Builds the one-step training loss and intermediate tensors."""
        x, physics_inputs, provided_measurement = self._extract_batch_inputs(batch)
        if x.ndim != 4:
            raise ValueError("x must have shape (B, C, H, W).")
        if not torch.is_floating_point(x):
            raise ValueError("x must be a floating-point tensor.")

        with _disable_cuda_autocast(x.device):
            x = x.to(dtype=torch.float32)
            physics_inputs = {
                key: value.to(device=x.device, dtype=torch.float32)
                for key, value in physics_inputs.items()
            }
            batch_size = x.shape[0]
            params = self.physics.sample_params(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
                **physics_inputs,
            )
            t, r = sample_t_r(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
            )
            clean_state_fn = getattr(self.physics, "clean_state", None)
            measure_state_fn = getattr(self.physics, "measure_state", None)
            null_state_fn = getattr(self.physics, "null_initializer_state", None)
            use_kspace_state_path = (
                callable(clean_state_fn)
                and callable(measure_state_fn)
                and callable(null_state_fn)
            )

            if use_kspace_state_path:
                # ── Full k-space path (MRI) ─────────────────────────────────────
                # Algorithm 1 state construction:
                #   x_k    = F C x_img
                #   y      = S x_k
                #   null_e = A†y + (I - A†A)e
                #   z      = (1 - t)x_k + t null_e
                kspace_x = clean_state_fn(x, params)
                if provided_measurement is not None:
                    if "mask" not in physics_inputs:
                        raise ValueError(
                            "batches with 'y_hat' must also include a matching 'mask'."
                        )
                    y = provided_measurement.to(device=x.device, dtype=x.dtype)
                else:
                    y = measure_state_fn(kspace_x, params)
                e = torch.randn_like(kspace_x).mul_(self.mri_noise_sigma)
                null_e = null_state_fn(y, e, params)
                t5 = _reshape_like_kspace(t)
                z = (1.0 - t5) * kspace_x + t5 * null_e

                kspace_vel_fn = self._make_kspace_velocity_fn(params)
                dudt = self._compute_detached_dudt(z, r, t, velocity_fn=kspace_vel_fn)
                u, v = kspace_vel_fn(z, r, t)

                V = u + _reshape_like_kspace(t - r) * dudt
                # target is naturally 0 on sampled columns (S̄·(e - kspace_x))
                target = null_e - kspace_x
                masked_V = V
                masked_v = v
                masked_target = target
                loss_mask = torch.ones_like(target)
                loss_u = torch.mean((masked_V - masked_target) ** 2)
                loss_v = torch.mean((masked_v - masked_target) ** 2)
                # Convert prediction back to image space for optional perceptual losses.
                prediction = self.physics.state_to_model_input(z - t5 * u, params)

            else:
                # ── Image-space path (inpainting / generic) ─────────────────────
                if provided_measurement is not None:
                    if "mask" not in physics_inputs:
                        raise ValueError(
                            "batches with 'y_hat' must also include a matching 'mask'."
                        )
                    y = provided_measurement.to(device=x.device, dtype=x.dtype)
                else:
                    y = self.physics.measure(x, params)
                e = torch.randn(x.shape, device=x.device, dtype=x.dtype)
                null_e = self.physics.null_initializer(y=y, e=e, params=params)
                t_image = _reshape_like_image(t)
                time_delta = _reshape_like_image(t - r)
                z = (1.0 - t_image) * x + t_image * null_e
                dudt = self._compute_detached_dudt(z, r, t)
                u, v = self._velocities(z, r, t)
                V = u + time_delta * dudt
                target = null_e - x
                loss_mask = torch.ones_like(target)
                if self.project_loss_to_measurement_space:
                    project_fn = getattr(self.physics, "project_for_loss", None)
                    if callable(project_fn):
                        masked_V = project_fn(V, params)
                        masked_v = project_fn(v, params)
                        masked_target = project_fn(target, params)
                        loss_mask = torch.ones_like(masked_target)
                    else:
                        loss_mask = _missing_region_mask(params)
                        masked_V = loss_mask * V
                        masked_v = loss_mask * v
                        masked_target = loss_mask * target
                else:
                    masked_V = V
                    masked_v = v
                    masked_target = target
                loss_u = torch.mean((masked_V - masked_target) ** 2)
                loss_v = torch.mean((masked_v - masked_target) ** 2)
                prediction = z - t_image * u
                if self.project_loss_to_measurement_space:
                    blend_fn = getattr(self.physics, "blend_reconstruction", None)
                    if callable(blend_fn):
                        prediction = blend_fn(prediction, x, params)

        aux_loss_lpips = torch.zeros((), device=x.device, dtype=x.dtype)
        aux_loss_convnext = torch.zeros((), device=x.device, dtype=x.dtype)
        loss_mse_reconstruction = torch.zeros((), device=x.device, dtype=x.dtype)
        perceptual_mask = (t < self.perceptual_max_t).to(dtype=x.dtype)

        if self.reconstruction_mse_lambda > 0:
            loss_mse_reconstruction = torch.mean((prediction - x) ** 2)
        if self._lpips_loss is not None:
            lpips_scores = self._lpips_loss(prediction, x).to(device=x.device, dtype=x.dtype)
            aux_loss_lpips = torch.mean(lpips_scores * perceptual_mask)
        if self._convnext_loss is not None:
            convnext_scores = self._convnext_loss(prediction, x).to(
                device=x.device,
                dtype=x.dtype,
            )
            aux_loss_convnext = torch.mean(convnext_scores * perceptual_mask)

        aux_loss = (
            self.lpips_lambda * aux_loss_lpips
            + self.convnext_lambda * aux_loss_convnext
            + self.reconstruction_mse_lambda * loss_mse_reconstruction
        )
        loss = loss_u + loss_v + aux_loss

        return {
            "loss": loss,
            "loss_u": loss_u,
            "loss_v": loss_v,
            "aux_loss": aux_loss,
            "aux_loss_lpips": aux_loss_lpips,
            "aux_loss_convnext": aux_loss_convnext,
            "loss_mse_reconstruction": loss_mse_reconstruction,
            "params": params,
            "y": y,
            "e": e,
            "t": t,
            "r": r,
            "null_e": null_e,
            "z": z,
            "u": u,
            "v": v,
            "dudt": dudt,
            "V": V,
            "prediction": prediction,
            "loss_mask": loss_mask,
            "masked_V": masked_V,
            "masked_v": masked_v,
            "target": target,
            "masked_target": masked_target,
        }


class IMFLoss:
    """Builds the null-space improved MeanFlow training loss.

    Implements:
        e      = randn_like(x)
        null_e = A†y + proj_null(e)
        z      = (1 - t)x + t null_e
        v      = fn(z, t, t)
        u,dudt = jvp(fn, (z, r, t), (v, 0, 1))
        V      = u + (t - r) stopgrad(dudt)
        loss   = metric(V, null_e - x)

    The JVP branch is evaluated on detached tensors so ``dudt`` acts strictly as
    a stop-gradient correction and does not keep a second model graph alive.
    """

    def __init__(
        self,
        model: PMFFlowModel,
        physics: DeepInvInpainting,
    ) -> None:
        """Initializes the reusable IMF loss builder."""
        self.model = model
        self.physics = physics
        self._velocity = _get_imf_velocity(model)

    def _compute_detached_dudt(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes IMF ``v`` and ``dudt`` outside the main autograd graph."""
        with torch.no_grad():
            z_detached = z.detach()
            r_detached = r.detach()
            t_detached = t.detach()
            v_seed = self._velocity(z_detached, t_detached, t_detached).detach()
            _, dudt = torch.func.jvp(
                self._velocity,
                (z_detached, r_detached, t_detached),
                (
                    v_seed,
                    torch.zeros_like(r_detached),
                    torch.ones_like(t_detached),
                ),
            )
        return v_seed, dudt.detach()

    def _extract_batch_inputs(
        self,
        batch: torch.Tensor | Mapping[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Returns the clean image plus optional physics inputs."""
        if torch.is_tensor(batch):
            return batch, {}, None
        x = batch.get("x")
        if not torch.is_tensor(x):
            raise ValueError("mapping batches must contain a tensor 'x'.")
        extra_inputs: dict[str, torch.Tensor] = {}
        mask = batch.get("mask")
        if torch.is_tensor(mask):
            extra_inputs["mask"] = mask
        smps = batch.get("smps")
        if torch.is_tensor(smps):
            extra_inputs["smps"] = smps
        y_hat = batch.get("y_hat")
        if y_hat is not None and not torch.is_tensor(y_hat):
            raise ValueError("mapping batches must use a tensor 'y_hat' when provided.")
        return x, extra_inputs, y_hat

    def __call__(
        self,
        batch: torch.Tensor | Mapping[str, object],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Builds the IMF loss and intermediate tensors."""
        x, physics_inputs, provided_measurement = self._extract_batch_inputs(batch)
        if x.ndim != 4:
            raise ValueError("x must have shape (B, C, H, W).")
        if not torch.is_floating_point(x):
            raise ValueError("x must be a floating-point tensor.")

        with _disable_cuda_autocast(x.device):
            x = x.to(dtype=torch.float32)
            physics_inputs = {
                key: value.to(device=x.device, dtype=torch.float32)
                for key, value in physics_inputs.items()
            }
            batch_size = x.shape[0]
            params = self.physics.sample_params(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
                **physics_inputs,
            )
            t, r = sample_t_r(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
            )
            if provided_measurement is not None:
                y = provided_measurement.to(device=x.device, dtype=x.dtype)
            else:
                y = self.physics.measure(x, params)
            e = torch.randn_like(x)
            null_e = self.physics.null_initializer(y=y, e=e, params=params)
            t_image = _reshape_like_image(t)
            z = (1.0 - t_image) * x + t_image * null_e

            v, dudt = self._compute_detached_dudt(z, r, t)
            u = self._velocity(z, r, t)
            V = u + _reshape_like_image(t - r) * dudt
            target = null_e - x
            loss_imf = torch.mean((V - target) ** 2)

        return {
            "loss": loss_imf,
            "loss_imf": loss_imf,
            "params": params,
            "y": y,
            "e": e,
            "t": t,
            "r": r,
            "null_e": null_e,
            "z": z,
            "imf_u": u,
            "imf_v_seed": v,
            "imf_dudt": dudt,
            "imf_V": V,
            "imf_target": target,
        }


def compute_training_loss(
    model: PMFFlowModel,
    physics: DeepInvInpainting,
    x: torch.Tensor | Mapping[str, object],
    *,
    project_loss_to_measurement_space: bool = True,
    lpips: bool = False,
    lpips_lambda: float = 1.0,
    convnext: bool = False,
    convnext_lambda: float = 0.0,
    reconstruction_mse_lambda: float = 0.0,
    perceptual_max_t: float = 1.0,
    mri_noise_sigma: float = 1.0,
    value_range: tuple[float, float] = (-1.0, 1.0),
    lpips_model: nn.Module | None = None,
    convnext_model: nn.Module | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Builds the one-step training loss and intermediate tensors.

    Args:
        model: Reconstruction model for the current experiment.
        physics: Inpainting physics wrapper.
        x: Clean image batch or one mapping batch with preprocessed MRI tensors.
        reconstruction_mse_lambda: Optional weight for the image-domain
            ``mean((prediction - x)^2)`` auxiliary regularizer.
        mri_noise_sigma: Std of the Gaussian noise injected into missing
            k-space columns in the MRI Algorithm 1 path.

    Returns:
        A dictionary containing the scalar loss and the intermediate tensors
        needed by later training and logging code.
    """
    return OneStepLoss(
        model=model,
        physics=physics,
        project_loss_to_measurement_space=project_loss_to_measurement_space,
        lpips=lpips,
        lpips_lambda=lpips_lambda,
        convnext=convnext,
        convnext_lambda=convnext_lambda,
        reconstruction_mse_lambda=reconstruction_mse_lambda,
        perceptual_max_t=perceptual_max_t,
        mri_noise_sigma=mri_noise_sigma,
        value_range=value_range,
        lpips_model=lpips_model,
        convnext_model=convnext_model,
    )(x)


def compute_imf_training_loss(
    model: PMFFlowModel,
    physics: DeepInvInpainting,
    x: torch.Tensor | Mapping[str, object],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Builds the null-space improved MeanFlow training loss."""
    return IMFLoss(
        model=model,
        physics=physics,
    )(x)
