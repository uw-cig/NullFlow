"""Evaluation helpers for one-step inpainting reconstruction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from NullFlow.metrics.reconstruction import ReconstructionMetrics
from NullFlow.model.pmf_flow import PMFFlowModel
from NullFlow.utils.config import ExperimentConfig


@dataclass
class ValidationVisuals:
    """Stores the fixed validation images used for qualitative logging."""

    ground_truth: torch.Tensor
    measurement: torch.Tensor
    null_e: torch.Tensor
    reconstruction: torch.Tensor


@dataclass
class ValidationResult:
    """Stores aggregated validation scalars and optional visualization batches."""

    scalars: dict[str, float]
    visuals: ValidationVisuals | None
    metadata: dict[str, object] = field(default_factory=dict)


def _rng_devices(device: torch.device) -> list[int]:
    """Returns CUDA device indices tracked by ``torch.random.fork_rng``."""
    if device.type != "cuda":
        return []
    if device.index is None:
        return list(range(torch.cuda.device_count()))
    return [device.index]


def _extract_batch_inputs(
    batch: torch.Tensor | Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, dict[str, object]]:
    """Returns the image tensor, physics inputs, and sample metadata."""
    if torch.is_tensor(batch):
        return batch, {}, None, {}

    x = batch.get("x")
    if not torch.is_tensor(x):
        raise ValueError("validation mapping batches must contain a tensor 'x'.")

    physics_inputs: dict[str, torch.Tensor] = {}
    smps = batch.get("smps")
    if torch.is_tensor(smps):
        physics_inputs["smps"] = smps
    mask = batch.get("mask")
    if torch.is_tensor(mask):
        physics_inputs["mask"] = mask
    y_hat = batch.get("y_hat")
    if y_hat is not None and not torch.is_tensor(y_hat):
        raise ValueError("validation mapping batches must use a tensor 'y_hat' when provided.")

    metadata: dict[str, object] = {}
    for key in ("file_stem", "volume_index", "slice_index"):
        if key in batch:
            metadata[key] = batch[key]
    return x, physics_inputs, y_hat, metadata


def _sample_params(
    physics: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None,
    physics_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Builds deterministic validation parameters for one batch."""
    if seed is None:
        return physics.sample_validation_params(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            **physics_inputs,
        )
    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return physics.sample_validation_params(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            **physics_inputs,
        )


def _sample_noise(
    reference: torch.Tensor,
    *,
    base_seed: int,
    sample_offset: int,
) -> torch.Tensor:
    """Builds deterministic validation noise for one image batch."""
    samples = []
    for index in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(base_seed + sample_offset + index)
        samples.append(
            torch.randn(
                reference[index].shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        )
    return torch.stack(samples, dim=0)


def _build_null_e(
    physics: Any,
    y: torch.Tensor,
    e: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Builds the one-step null-space initialization tensor."""
    return physics.null_initializer(y=y, e=e, params=params)


def _reconstruct(
    model: PMFFlowModel,
    null_e: torch.Tensor,
    *,
    sampling_steps: int = 1,
    sampling_method: str = "meanflow_update",
    sampling_objective: str = "pmf",
) -> torch.Tensor:
    """Runs one-step or multi-step sampling from ``null_e``."""
    if sampling_objective == "imf":
        return _imf_reconstruct(model, null_e)
    if sampling_objective != "pmf":
        raise ValueError("sampling_objective must be one of: pmf, imf.")
    return multistep_sample(
        model,
        null_e,
        sampling_steps,
        method=sampling_method,
    )


def _imf_reconstruct(
    model: PMFFlowModel,
    null_e: torch.Tensor,
) -> torch.Tensor:
    """Runs IMF one-step sampling: ``x = null_e - fn(null_e, r=0, t=1)``."""
    batch_size = null_e.shape[0]
    r = torch.zeros(batch_size, device=null_e.device, dtype=null_e.dtype)
    t = torch.ones(batch_size, device=null_e.device, dtype=null_e.dtype)
    imf_velocity_fn = getattr(model, "predict_imf_velocity", None)
    if callable(imf_velocity_fn) and not hasattr(model, "module"):
        u = imf_velocity_fn(null_e, r, t)
    else:
        u, _ = model(null_e, r, t, return_velocity_pair=True)
    return null_e - u


def _blend_reconstruction(
    physics: Any,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Builds the visualization/metric reconstruction for the active physics."""
    blend_fn = getattr(physics, "blend_reconstruction", None)
    if callable(blend_fn):
        return blend_fn(prediction, ground_truth, params)
    return prediction


def _measurement_fill_image(
    prediction: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fills missing image-space entries with the model prediction."""
    if "mask" not in params:
        raise ValueError("measurement_fill requires params['mask'].")
    mask = params["mask"]
    return y + (1.0 - mask) * prediction


def _measurement_fill_mri(
    physics: Any,
    prediction: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fills missing MRI k-space entries and maps the result to image space."""
    model_output_to_state = getattr(physics, "model_output_to_state", None)
    state_to_model_input = getattr(physics, "state_to_model_input", None)
    zero_fill_state = getattr(physics, "zero_fill_state", None)
    if (
        not callable(model_output_to_state)
        or not callable(state_to_model_input)
        or not callable(zero_fill_state)
    ):
        raise ValueError(
            "MRI measurement_fill requires model_output_to_state, "
            "state_to_model_input, and zero_fill_state helpers."
        )
    if "mask" not in params:
        raise ValueError("MRI measurement_fill requires params['mask'].")

    x_hat_k = model_output_to_state(prediction, params)
    observed_k = zero_fill_state(y, params)
    missing_mask = (1.0 - params["mask"]).unsqueeze(1)
    prediction_k = observed_k + missing_mask * x_hat_k
    return state_to_model_input(prediction_k, params)


def _select_reconstruction(
    *,
    reconstruction_mode: str,
    physics: Any,
    prediction: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
    ground_truth: torch.Tensor | None,
    is_kspace_state: bool,
) -> torch.Tensor:
    """Selects the configured reconstruction mode for validation or solving."""
    if reconstruction_mode == "model_only":
        if ground_truth is None:
            return prediction
        return _blend_reconstruction(physics, prediction, ground_truth, params)
    if reconstruction_mode == "measurement_fill":
        if is_kspace_state:
            return _measurement_fill_mri(physics, prediction, y, params)
        return _measurement_fill_image(prediction, y, params)
    raise ValueError(
        "reconstruction_mode must be one of: model_only, measurement_fill."
    )


def _append_visual_batch(
    storage: list[torch.Tensor],
    batch: torch.Tensor,
    *,
    limit: int,
) -> None:
    """Appends up to ``limit`` samples from one tensor batch."""
    current = sum(chunk.shape[0] for chunk in storage)
    remaining = limit - current
    if remaining <= 0:
        return
    storage.append(batch[:remaining].detach().cpu())


def _build_visuals(
    ground_truth: list[torch.Tensor],
    measurement: list[torch.Tensor],
    null_e: list[torch.Tensor],
    reconstruction: list[torch.Tensor],
    *,
    limit: int,
) -> ValidationVisuals | None:
    """Builds the qualitative validation payload from collected batches."""
    if not ground_truth:
        return None

    visuals = ValidationVisuals(
        ground_truth=torch.cat(ground_truth, dim=0),
        measurement=torch.cat(measurement, dim=0),
        null_e=torch.cat(null_e, dim=0),
        reconstruction=torch.cat(reconstruction, dim=0),
    )
    if visuals.ground_truth.shape[0] < limit:
        raise ValueError(
            "validation loader must provide at least visualization_subset_size samples."
        )
    return visuals


@torch.no_grad()
def multistep_sample(
    model: PMFFlowModel,
    null_e: torch.Tensor,
    n_steps: int,
    *,
    method: str = "meanflow_update",
) -> torch.Tensor:
    """Runs null-space MeanFlow multi-step sampling backward from t=1 to t=0.

    ``method='meanflow_update'`` partitions ``[1, 0]`` into ``n_steps``
    descending intervals and applies::

        z_r = z - ((t - r) / t) * (z - net(z, r, t))

    ``method='direct_prediction'`` uses the direct recurrent update::

        z = net(z, r, t)

    for each adjacent pair ``(t, r)`` until ``r=0``. When ``n_steps=1``, both
    methods reduce exactly to ``model(null_e, r=0, t=1)``.

    Args:
        model: A PMFFlowModel whose forward(z, r, t) returns a reconstruction.
        null_e: Initial state at t=1 with shape (B, C, H, W).
        n_steps: Number of model evaluations. Must be >= 1.
        method: Sampling update rule, either ``meanflow_update`` or
            ``direct_prediction``.

    Returns:
        Reconstruction estimate with the same shape as null_e.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if method not in {"meanflow_update", "direct_prediction"}:
        raise ValueError("method must be one of: meanflow_update, direct_prediction.")

    z = null_e.clone()
    time_grid = torch.linspace(
        1.0,
        0.0,
        n_steps + 1,
        device=z.device,
        dtype=z.dtype,
    )
    batch_size = z.shape[0]
    for index in range(n_steps):
        t = time_grid[index].expand(batch_size)
        r = time_grid[index + 1].expand(batch_size)
        x_hat = model(z, r=r, t=t)
        if method == "direct_prediction":
            z = x_hat
            continue
        coefficient = PMFFlowModel.reshape_time_like_image((t - r) / t)
        z = z - coefficient * (z - x_hat)

    return z


@torch.no_grad()
def solve_one_step(
    model: PMFFlowModel,
    physics: Any,
    y: torch.Tensor,
    e: torch.Tensor,
    params: dict[str, torch.Tensor],
    ground_truth: torch.Tensor | None = None,
    reconstruction_mode: str = "model_only",
    sampling_steps: int = 1,
    sampling_method: str = "meanflow_update",
    sampling_objective: str = "pmf",
) -> torch.Tensor:
    """Builds ``null_e`` and returns the one-step reconstruction."""
    null_state_fn = getattr(physics, "null_initializer_state", None)
    state_to_model_input = getattr(physics, "state_to_model_input", None)
    if callable(null_state_fn) and callable(state_to_model_input) and y.ndim == 5:
        null_e_k = null_state_fn(y, e, params)
        null_e_img = state_to_model_input(null_e_k, params)
        prediction = _reconstruct(
            model,
            null_e_img,
            sampling_steps=sampling_steps,
            sampling_method=sampling_method,
            sampling_objective=sampling_objective,
        )
        return _select_reconstruction(
            reconstruction_mode=reconstruction_mode,
            physics=physics,
            prediction=prediction,
            y=y,
            params=params,
            ground_truth=ground_truth,
            is_kspace_state=True,
        )

    prediction = _reconstruct(
        model,
        _build_null_e(physics, y, e, params),
        sampling_steps=sampling_steps,
        sampling_method=sampling_method,
        sampling_objective=sampling_objective,
    )
    return _select_reconstruction(
        reconstruction_mode=reconstruction_mode,
        physics=physics,
        prediction=prediction,
        y=y,
        params=params,
        ground_truth=ground_truth,
        is_kspace_state=False,
    )


@torch.no_grad()
def run_validation(
    model: PMFFlowModel,
    physics: Any,
    val_loader: Iterable[torch.Tensor | Mapping[str, object]],
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    lpips_model: torch.nn.Module | None = None,
    fid_extractor: torch.nn.Module | None = None,
) -> ValidationResult:
    """Runs one validation pass and collects metrics plus fixed visuals."""
    if not config.evaluation.run_one_step_solver:
        raise ValueError("run_one_step_solver must be enabled for validation.")

    target_device = torch.device(device)
    metric_suite = ReconstructionMetrics(
        config.evaluation.metrics,
        device=target_device,
        value_range=config.data.normalization_range,
        fid_real_stats_path=config.evaluation.fid_real_stats_path,
        lpips_model=lpips_model,
        fid_extractor=fid_extractor,
    )
    visual_limit = config.evaluation.visualization_subset_size
    ground_truth_batches: list[torch.Tensor] = []
    measurement_batches: list[torch.Tensor] = []
    null_e_batches: list[torch.Tensor] = []
    reconstruction_batches: list[torch.Tensor] = []
    visual_mask_ids: list[str] = []
    visual_mask_sources: list[str] = []
    visual_file_stems: list[str] = []
    visual_volume_indices: list[int] = []
    visual_slice_indices: list[int] = []
    sample_offset = 0
    model_was_training = model.training
    model.eval()

    try:
        for batch in val_loader:
            x, physics_inputs, provided_measurement, batch_metadata = _extract_batch_inputs(batch)
            x = x.to(device=target_device)
            if x.ndim != 4:
                raise ValueError("validation batches must have shape (B, C, H, W).")
            if not torch.is_floating_point(x):
                raise ValueError("validation batches must be floating-point tensors.")

            batch_size = x.shape[0]
            visual_count_before = sum(chunk.shape[0] for chunk in ground_truth_batches)
            params = _sample_params(
                physics,
                batch_size=batch_size,
                device=target_device,
                dtype=x.dtype,
                seed=config.evaluation.fixed_mask_seed,
                physics_inputs={
                    key: value.to(device=target_device)
                    for key, value in physics_inputs.items()
                },
            )
            clean_state_fn = getattr(physics, "clean_state", None)
            measure_state_fn = getattr(physics, "measure_state", None)
            null_state_fn = getattr(physics, "null_initializer_state", None)
            state_to_model_input = getattr(physics, "state_to_model_input", None)
            use_kspace_state_path = (
                callable(clean_state_fn)
                and callable(measure_state_fn)
                and callable(null_state_fn)
                and callable(state_to_model_input)
            )

            if use_kspace_state_path:
                x_k = clean_state_fn(x, params)
                if provided_measurement is not None:
                    if "mask" not in physics_inputs:
                        raise ValueError(
                            "validation batches with 'y_hat' must also include a matching 'mask'."
                        )
                    y = provided_measurement.to(device=target_device, dtype=x.dtype)
                else:
                    y = measure_state_fn(x_k, params)
                e_k = _sample_noise(
                    x_k,
                    base_seed=config.evaluation.fixed_noise_seed,
                    sample_offset=sample_offset,
                )
                noise_sigma = config.data.mri.noise_sigma if config.data.mri is not None else 1.0
                e_k = e_k.mul_(noise_sigma)
                null_e_k = null_state_fn(y, e_k, params)
                null_e = state_to_model_input(null_e_k, params)
                prediction = _reconstruct(
                    model,
                    null_e,
                    sampling_steps=config.evaluation.sampling_steps,
                    sampling_method=config.evaluation.sampling_method,
                    sampling_objective=config.training.loss_method,
                )
                reconstruction = _select_reconstruction(
                    reconstruction_mode=config.evaluation.reconstruction_mode,
                    physics=physics,
                    prediction=prediction,
                    y=y,
                    params=params,
                    ground_truth=x,
                    is_kspace_state=True,
                )
                measurement_visual_fn = getattr(physics, "measurement_for_visualization", None)
                if callable(measurement_visual_fn):
                    measurement_visual = measurement_visual_fn(y, params)
                else:
                    measurement_visual = state_to_model_input(y, params)
            else:
                e = _sample_noise(
                    x,
                    base_seed=config.evaluation.fixed_noise_seed,
                    sample_offset=sample_offset,
                )
                if provided_measurement is not None:
                    if "mask" not in physics_inputs:
                        raise ValueError(
                            "validation batches with 'y_hat' must also include a matching 'mask'."
                        )
                    y = provided_measurement.to(device=target_device, dtype=x.dtype)
                else:
                    y = physics.measure(x, params)
                null_e = _build_null_e(physics, y, e, params)
                prediction = _reconstruct(
                    model,
                    null_e,
                    sampling_steps=config.evaluation.sampling_steps,
                    sampling_method=config.evaluation.sampling_method,
                    sampling_objective=config.training.loss_method,
                )
                reconstruction = _select_reconstruction(
                    reconstruction_mode=config.evaluation.reconstruction_mode,
                    physics=physics,
                    prediction=prediction,
                    y=y,
                    params=params,
                    ground_truth=x,
                    is_kspace_state=False,
                )
                measurement_visual_fn = getattr(physics, "measurement_for_visualization", None)
                if callable(measurement_visual_fn):
                    measurement_visual = measurement_visual_fn(y, params)
                else:
                    measurement_visual = y

            metric_suite.update(reconstruction, x)
            _append_visual_batch(ground_truth_batches, x, limit=visual_limit)
            _append_visual_batch(measurement_batches, measurement_visual, limit=visual_limit)
            _append_visual_batch(null_e_batches, null_e, limit=visual_limit)
            _append_visual_batch(reconstruction_batches, reconstruction, limit=visual_limit)

            metadata_limit = min(batch_size, max(visual_limit - visual_count_before, 0))
            if metadata_limit > 0:
                mask_metadata_fn = getattr(physics, "resolve_mask_metadata", None)
                if callable(mask_metadata_fn):
                    mask_metadata = mask_metadata_fn(params)
                    visual_mask_ids.extend(list(mask_metadata.get("mask_ids", ())[:metadata_limit]))
                    visual_mask_sources.extend(
                        list(mask_metadata.get("mask_sources", ())[:metadata_limit])
                    )
                if "file_stem" in batch_metadata:
                    stems = batch_metadata["file_stem"]
                    if isinstance(stems, (list, tuple)):
                        visual_file_stems.extend([str(item) for item in stems[:metadata_limit]])
                    elif isinstance(stems, str) and metadata_limit > 0:
                        visual_file_stems.append(stems)
                if "volume_index" in batch_metadata:
                    values = batch_metadata["volume_index"]
                    if torch.is_tensor(values):
                        visual_volume_indices.extend(values[:metadata_limit].detach().cpu().tolist())
                    elif isinstance(values, int) and metadata_limit > 0:
                        visual_volume_indices.append(values)
                if "slice_index" in batch_metadata:
                    values = batch_metadata["slice_index"]
                    if torch.is_tensor(values):
                        visual_slice_indices.extend(values[:metadata_limit].detach().cpu().tolist())
                    elif isinstance(values, int) and metadata_limit > 0:
                        visual_slice_indices.append(values)
            sample_offset += batch_size
    finally:
        if model_was_training:
            model.train()

    scalars = {
        f"val_{name}": value
        for name, value in metric_suite.compute().items()
    }
    visuals = _build_visuals(
        ground_truth_batches,
        measurement_batches,
        null_e_batches,
        reconstruction_batches,
        limit=visual_limit,
    )
    metadata: dict[str, object] = {
        "fixed_noise_seed": config.evaluation.fixed_noise_seed,
        "fixed_mask_seed": config.evaluation.fixed_mask_seed,
    }
    if visual_mask_ids:
        metadata["visual_mask_ids"] = tuple(visual_mask_ids[:visual_limit])
    if visual_mask_sources:
        metadata["visual_mask_sources"] = tuple(visual_mask_sources[:visual_limit])
    if visual_file_stems:
        metadata["visual_file_stems"] = tuple(visual_file_stems[:visual_limit])
    if visual_volume_indices:
        metadata["visual_volume_indices"] = tuple(visual_volume_indices[:visual_limit])
    if visual_slice_indices:
        metadata["visual_slice_indices"] = tuple(visual_slice_indices[:visual_limit])
    return ValidationResult(
        scalars=scalars,
        visuals=visuals,
        metadata=metadata,
    )
