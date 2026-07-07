"""Epoch-based training engine helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import faulthandler
import json
import os
from pathlib import Path
from sys import stderr
from time import perf_counter, sleep
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from NullFlow.data import build_datasets, safe_collate
from NullFlow.evaluation.inpainting import ValidationResult, run_validation
from NullFlow.inverse import build_physics
from NullFlow.model.pmf_adapter import build_pmf_model
from NullFlow.training.checkpoint import load_from_config, save_checkpoint
from NullFlow.training.losses import IMFLoss, OneStepLoss
from NullFlow.training.step import TrainStepResult, run_train_step
from NullFlow.training.wandb_logger import WandbLogger
from NullFlow.utils.config import ExperimentConfig
from NullFlow.utils.seed import make_torch_generator, seed_worker, set_seed


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FALSEY_ENV_VALUES = {"", "0", "false", "False", "no", "No", "off", "Off"}


@dataclass
class EngineResult:
    """Stores the final training loop state."""

    start_epoch: int
    completed_epochs: int
    global_step: int
    checkpoint_paths: tuple[Path, ...]
    last_validation: ValidationResult | None


@dataclass
class DistributedState:
    """Stores distributed runtime information for one process."""

    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main_process: bool = True
    initialized_here: bool = False


@dataclass
class ValidationThresholdState:
    """Stores one-shot validation-threshold optimizer adjustment state."""

    consecutive_hits: int = 0
    triggered: bool = False
    adjustment_count: int = 0
    trigger_epoch: int = 0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ValidationThresholdState":
        """Builds one threshold state from checkpoint payload data."""
        if payload is None:
            return cls()
        return cls(
            consecutive_hits=int(payload.get("consecutive_hits", 0)),
            triggered=bool(payload.get("triggered", False)),
            adjustment_count=int(payload.get("adjustment_count", 0)),
            trigger_epoch=int(payload.get("trigger_epoch", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the threshold state for checkpoints or broadcasts."""
        return {
            "consecutive_hits": self.consecutive_hits,
            "triggered": self.triggered,
            "adjustment_count": self.adjustment_count,
            "trigger_epoch": self.trigger_epoch,
        }


def _set_init_seed(config: ExperimentConfig) -> None:
    """Builds identical model initialization across all ranks."""
    set_seed(config.seed)


def _set_runtime_seed(
    config: ExperimentConfig,
    state: DistributedState,
) -> None:
    """Offsets the training-time RNG by rank after model construction."""
    if not state.enabled:
        return
    torch.manual_seed(config.seed + state.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + state.rank)
        torch.cuda.manual_seed_all(config.seed + state.rank)


def _resolve_device(device: torch.device | str) -> torch.device:
    """Normalizes a device specification."""
    return torch.device(device)


def _resolve_output_dir(path_value: str) -> Path:
    """Resolves an output directory relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _debug_progress_enabled() -> bool:
    """Checks whether terminal progress diagnostics should be printed."""
    return os.environ.get("ONESTEP_TRAIN_DEBUG", "0") not in _FALSEY_ENV_VALUES


def _debug_all_ranks_enabled() -> bool:
    """Checks whether verbose per-rank progress diagnostics are enabled."""
    return os.environ.get("ONESTEP_TRAIN_DEBUG_ALL_RANKS", "1") not in _FALSEY_ENV_VALUES


def _debug_step_interval() -> int:
    """Returns the optimizer-step interval for terminal diagnostics."""
    raw_value = os.environ.get("ONESTEP_TRAIN_DEBUG_STEP_INTERVAL", "1")
    try:
        interval = int(raw_value)
    except ValueError:
        return 1
    return max(interval, 1)


def _maybe_len(value: Any) -> int | None:
    """Returns ``len(value)`` when available without raising."""
    try:
        return len(value)
    except TypeError:
        return None


def _debug_print(
    state: DistributedState,
    message: str,
    *,
    main_process: bool,
    every_rank: bool = False,
) -> None:
    """Prints one flushed terminal diagnostic when debug progress is enabled."""
    if not _debug_progress_enabled():
        return
    if not every_rank and not main_process:
        return
    if every_rank and state.enabled and not _debug_all_ranks_enabled():
        if not main_process:
            return
    rank_label = f"rank {state.rank}/{state.world_size}" if state.enabled else "single"
    print(
        f"[train-debug][{rank_label}][pid {os.getpid()}] {message}",
        file=stderr,
        flush=True,
    )


def _early_debug_print(message: str) -> None:
    """Prints flushed diagnostics before distributed state exists."""
    if not _debug_progress_enabled():
        return
    rank = os.environ.get("RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        (
            f"[train-debug][rank {rank}/{world_size} local_rank {local_rank}]"
            f"[pid {os.getpid()}] {message}"
        ),
        file=stderr,
        flush=True,
    )


def _append_timestamp(path_value: str, timestamp: str) -> str:
    """Appends one timestamp suffix to the last path component."""
    path = Path(path_value).expanduser()
    parent = path.parent
    stamped_name = f"{path.name}_{timestamp}"
    return str(parent / stamped_name)


def _shared_run_token() -> str:
    """Builds one filesystem-safe identifier shared across torchrun ranks."""
    raw_token = (
        os.environ.get("ONESTEP_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or f"{os.environ.get('MASTER_ADDR', 'local')}_{os.environ.get('MASTER_PORT', 'single')}"
    )
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in raw_token
    )
    return sanitized or "default"


def _shared_timestamp_file(path_value: str) -> Path:
    """Returns the per-run timestamp coordination file."""
    base_path = Path(path_value).expanduser()
    return (
        base_path.parent
        / ".onestep_sync"
        / f"{base_path.name}_{_shared_run_token()}_timestamp.txt"
    )


def _write_text_atomic(path: Path, content: str) -> None:
    """Writes a small text payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _read_shared_timestamp(path: Path) -> str | None:
    """Reads one previously published shared timestamp when available."""
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return content or None


def _resolve_shared_timestamp(
    config: ExperimentConfig,
    state: DistributedState,
) -> str:
    """Returns one timestamp string shared across all ranks without collectives."""
    env_timestamp = os.environ.get("ONESTEP_RUN_TIMESTAMP")
    if env_timestamp:
        return env_timestamp

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not state.enabled:
        return timestamp

    timestamp_file = _shared_timestamp_file(config.output_dir)
    if state.is_main_process:
        _write_text_atomic(timestamp_file, timestamp)
        return timestamp

    timeout_s = 120.0
    deadline = perf_counter() + timeout_s
    while perf_counter() < deadline:
        shared_timestamp = _read_shared_timestamp(timestamp_file)
        if shared_timestamp is not None:
            return shared_timestamp
        sleep(0.1)
    raise TimeoutError(
        "Timed out while waiting for the main rank to publish the shared output_dir timestamp."
    )


def _validation_threshold_sync_path(output_dir: str, epoch: int) -> Path:
    """Returns the synchronization payload path for one validation epoch."""
    return _resolve_output_dir(output_dir) / ".onestep_sync" / f"validation_threshold_epoch_{epoch:04d}.json"


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """Reads one JSON mapping when the file exists."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return dict(json.loads(raw_text))


def _stamp_output_dir(
    config: ExperimentConfig,
    state: DistributedState,
) -> ExperimentConfig:
    """Adds one shared timestamp suffix to output_dir for each training launch."""
    timestamp = _resolve_shared_timestamp(config, state)
    return config.with_updates(
        output_dir=_append_timestamp(config.output_dir, timestamp),
    )


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    shuffle: bool,
    seed: int,
    sampler: DistributedSampler | None = None,
) -> DataLoader[torch.Tensor]:
    """Builds a reproducible data loader for one split."""
    use_persistent_workers = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        drop_last=False,
        sampler=sampler,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
        collate_fn=safe_collate,
    )


def _build_optimizer(
    config: ExperimentConfig,
    model: torch.nn.Module,
) -> Optimizer:
    """Builds the default optimizer for PMF training."""
    return AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
    )


def _build_lr_scheduler(
    config: ExperimentConfig,
    optimizer: Optimizer,
) -> LRScheduler | None:
    """Builds the configured learning-rate scheduler when enabled."""
    scheduler_type = config.training.lr_scheduler_type
    if scheduler_type == "none":
        return None
    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config.training.lr_scheduler_step_size,
            gamma=config.training.lr_scheduler_gamma,
        )
    raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")


def _current_lrs(optimizer: Optimizer) -> tuple[float, ...]:
    """Returns the optimizer learning rates for checkpointing and broadcasts."""
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def _set_optimizer_lrs(
    optimizer: Optimizer,
    lrs: tuple[float, ...] | list[float],
) -> None:
    """Applies one learning-rate tuple to all optimizer parameter groups."""
    if len(lrs) != len(optimizer.param_groups):
        raise ValueError("learning-rate payload must match optimizer param_groups.")
    for group, lr in zip(optimizer.param_groups, lrs, strict=True):
        group["lr"] = float(lr)


def _apply_validation_threshold(
    config: ExperimentConfig,
    optimizer: Optimizer,
    validation_scalars: dict[str, float],
    threshold_state: ValidationThresholdState,
    *,
    epoch: int,
) -> tuple[ValidationThresholdState, dict[str, float], bool]:
    """Applies the configured one-shot optimizer adjustment after validation."""
    if not config.training.validation_threshold_enabled:
        return threshold_state, {}, False

    metric_name = config.training.validation_monitor_metric
    if metric_name not in validation_scalars:
        raise ValueError(
            f"validation_threshold_enabled requires the metric '{metric_name}' in validation scalars."
        )

    monitored_value = float(validation_scalars[metric_name])
    mode = config.training.validation_metric_mode
    threshold = config.training.validation_threshold_value
    hit = monitored_value >= threshold if mode == "max" else monitored_value <= threshold

    next_state = ValidationThresholdState(
        consecutive_hits=threshold_state.consecutive_hits,
        triggered=threshold_state.triggered,
        adjustment_count=threshold_state.adjustment_count,
        trigger_epoch=threshold_state.trigger_epoch,
    )
    if next_state.triggered:
        next_state.consecutive_hits = (
            next_state.consecutive_hits + 1 if hit else 0
        )
        return next_state, {
            "train_validation_threshold_hits": float(next_state.consecutive_hits),
            "train_validation_threshold_triggered": 0.0,
            "train_optimizer_adjustment_count": float(next_state.adjustment_count),
        }, False

    next_state.consecutive_hits = next_state.consecutive_hits + 1 if hit else 0
    triggered_now = next_state.consecutive_hits >= config.training.validation_threshold_patience
    if triggered_now:
        scaled_lrs = tuple(
            lr * config.training.validation_threshold_lr_scale
            for lr in _current_lrs(optimizer)
        )
        _set_optimizer_lrs(optimizer, scaled_lrs)
        if config.training.validation_threshold_reset_optimizer_state:
            optimizer.state.clear()
        next_state.triggered = True
        next_state.adjustment_count += 1
        next_state.trigger_epoch = epoch

    scalars = {
        "train_validation_threshold_hits": float(next_state.consecutive_hits),
        "train_validation_threshold_triggered": 1.0 if triggered_now else 0.0,
        "train_optimizer_adjustment_count": float(next_state.adjustment_count),
    }
    if triggered_now:
        scalars["train_lr"] = float(_current_lrs(optimizer)[0])
    return next_state, scalars, triggered_now


def _sync_validation_threshold_state(
    *,
    optimizer: Optimizer,
    threshold_state: ValidationThresholdState,
    triggered_now: bool,
    distributed_state: DistributedState,
    is_main_process: bool,
    sync_path: Path,
) -> ValidationThresholdState:
    """Synchronizes threshold state and optimizer LR updates through one shared file."""
    if is_main_process:
        payload = {
            "threshold_state": threshold_state.to_dict(),
            "triggered_now": bool(triggered_now),
            "lrs": list(_current_lrs(optimizer)),
        }
        _write_text_atomic(
            sync_path,
            json.dumps(payload, sort_keys=True),
        )
        return threshold_state

    timeout_s = 600.0
    deadline = perf_counter() + timeout_s
    synced_payload: dict[str, Any] | None = None
    while perf_counter() < deadline:
        synced_payload = _read_json_file(sync_path)
        if synced_payload is not None:
            break
        sleep(0.1)
    if synced_payload is None:
        raise TimeoutError(
            f"Timed out while waiting for validation-threshold sync payload: {sync_path}"
        )

    synced_state = ValidationThresholdState.from_mapping(
        dict(synced_payload.get("threshold_state", {}))
    )
    synced_triggered_now = bool(synced_payload.get("triggered_now", False))
    synced_lrs = tuple(float(value) for value in synced_payload.get("lrs", ()))
    if not is_main_process:
        _set_optimizer_lrs(optimizer, synced_lrs)
        if synced_triggered_now:
            optimizer.state.clear()
    return synced_state


def _train_minibatch_size(config: ExperimentConfig) -> int:
    """Returns the per-forward minibatch size used by the train loader."""
    return (
        config.training.batch_size
        if config.training.minibatch_size is None
        else config.training.minibatch_size
    )


def _accumulation_steps(config: ExperimentConfig) -> int:
    """Returns the number of train minibatches per optimizer update."""
    return config.training.batch_size // _train_minibatch_size(config)


def _build_scaler(
    config: ExperimentConfig,
    device: torch.device,
) -> torch.amp.GradScaler | None:
    """Builds the default AMP scaler when CUDA mixed precision is enabled."""
    if not config.training.amp_enabled or device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")


def _env_int(name: str) -> int:
    """Reads one integer environment variable required by DDP."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise ValueError(
            f"training.distributed requires the '{name}' environment variable. "
            "Launch with torchrun to populate rank metadata."
        )
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"Environment variable '{name}' must be an integer.") from error


def _setup_distributed(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.device, DistributedState]:
    """Initializes DDP state from the current environment when enabled."""
    _early_debug_print(
        f"distributed setup enter enabled={config.training.distributed} device={device}"
    )
    if not config.training.distributed:
        _early_debug_print("distributed setup skipped")
        return device, DistributedState(enabled=False)
    if not dist.is_available():
        raise ValueError("torch.distributed is not available in this environment.")

    rank = _env_int("RANK")
    world_size = _env_int("WORLD_SIZE")
    local_rank = _env_int("LOCAL_RANK")
    _early_debug_print(
        f"distributed env ready rank={rank} world_size={world_size} local_rank={local_rank}"
    )
    if world_size <= 1:
        raise ValueError("training.distributed requires WORLD_SIZE > 1.")

    target_device = device
    if device.type == "cuda":
        _early_debug_print(f"cuda set_device start cuda:{local_rank}")
        target_device = torch.device("cuda", local_rank)
        torch.cuda.set_device(target_device)
        _early_debug_print(f"cuda set_device done cuda:{local_rank}")
    elif config.training.backend == "nccl":
        raise ValueError("The NCCL backend requires a CUDA device.")

    initialized_here = False
    if not dist.is_initialized():
        _early_debug_print(f"init_process_group start backend={config.training.backend}")
        dist.init_process_group(backend=config.training.backend)
        initialized_here = True
        _early_debug_print(f"init_process_group done backend={config.training.backend}")
    else:
        _early_debug_print("process group already initialized")

    return target_device, DistributedState(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main_process=(rank == 0),
        initialized_here=initialized_here,
    )


def _cleanup_distributed(state: DistributedState) -> None:
    """Destroys the process group owned by the current engine instance."""
    if not state.initialized_here:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_train_sampler(
    dataset: Dataset,
    state: DistributedState,
) -> DistributedSampler | None:
    """Builds the DDP sampler for one training dataset when needed."""
    if not state.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=state.world_size,
        rank=state.rank,
        shuffle=True,
        drop_last=False,
    )


def _set_epoch(
    sampler: DistributedSampler | None,
    epoch: int,
) -> None:
    """Updates the distributed sampler seed for one epoch."""
    if sampler is None:
        return
    sampler.set_epoch(epoch)


def _wrap_model(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    device: torch.device,
    state: DistributedState,
) -> nn.Module:
    """Wraps the model for DDP when distributed training is enabled."""
    wrapped_model = model
    if state.enabled and config.training.sync_batchnorm:
        wrapped_model = nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)

    wrapped_model = wrapped_model.to(device=device)
    if not state.enabled:
        return wrapped_model
    if device.type == "cuda":
        ddp_model = DistributedDataParallel(
            wrapped_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            init_sync=False,
            find_unused_parameters=config.training.find_unused_parameters,
        )
    else:
        ddp_model = DistributedDataParallel(
            wrapped_model,
            broadcast_buffers=False,
            init_sync=False,
            find_unused_parameters=config.training.find_unused_parameters,
        )

    for name in (
        "predict_velocity_pair",
        "predict_head_pair",
        "set_conditioning",
        "clear_conditioning",
    ):
        if hasattr(ddp_model.module, name):
            setattr(ddp_model, name, getattr(ddp_model.module, name))
    return ddp_model


def _build_train_scalars(
    step_result: TrainStepResult,
    *,
    data_time: float,
    step_time: float,
) -> dict[str, float]:
    """Builds the scalar payload logged for one train step."""
    return {
        **step_result.scalars,
        "train_data_time": data_time,
        "train_step_time": step_time,
    }


def _should_run(index: int, every: int) -> bool:
    """Checks whether one interval-based hook should run."""
    return index > 0 and index % every == 0


def _checkpoint_path(config: ExperimentConfig, epoch: int) -> Path:
    """Builds the checkpoint path for one completed epoch."""
    output_dir = _resolve_output_dir(config.output_dir)
    return output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"


def run_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    is_main_process: bool = True,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
    model: torch.nn.Module | None = None,
    physics: Any | None = None,
    optimizer: Optimizer | None = None,
    loss_fn: OneStepLoss | Any | None = None,
    logger: WandbLogger | Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: LRScheduler | Any | None = None,
    lpips_model: torch.nn.Module | None = None,
    convnext_model: torch.nn.Module | None = None,
    fid_extractor: torch.nn.Module | None = None,
) -> EngineResult:
    """Runs epoch-based training, checkpointing, validation, and logging."""
    if _debug_progress_enabled():
        faulthandler.enable(all_threads=True)
        _early_debug_print("faulthandler enabled")
    target_device = _resolve_device(device)
    _early_debug_print(f"resolved requested device={target_device}")
    target_device, distributed_state = _setup_distributed(config, target_device)
    config = _stamp_output_dir(config, distributed_state)
    main_process = is_main_process and distributed_state.is_main_process
    _debug_print(
        distributed_state,
        f"run start device={target_device} output_dir={config.output_dir}",
        main_process=main_process,
        every_rank=True,
    )
    _set_init_seed(config)

    if train_dataset is None or val_dataset is None:
        _debug_print(
            distributed_state,
            "building datasets",
            main_process=main_process,
            every_rank=True,
        )
        built_train_dataset, built_val_dataset = build_datasets(config.data)
        if train_dataset is None:
            train_dataset = built_train_dataset
        if val_dataset is None:
            val_dataset = built_val_dataset
    _debug_print(
        distributed_state,
        f"datasets ready train_len={_maybe_len(train_dataset)} val_len={_maybe_len(val_dataset)}",
        main_process=main_process,
        every_rank=True,
    )

    train_sampler = _build_train_sampler(train_dataset, distributed_state)
    train_loader = _build_loader(
        train_dataset,
        batch_size=_train_minibatch_size(config),
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        shuffle=True,
        seed=config.seed,
        sampler=train_sampler,
    )
    val_loader = None
    if main_process:
        val_loader = _build_loader(
            val_dataset,
            batch_size=config.evaluation.val_batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers,
            shuffle=False,
            seed=config.seed + 1,
        )
    _debug_print(
        distributed_state,
        (
            "loaders ready "
            f"train_batches={_maybe_len(train_loader)} "
            f"val_batches={_maybe_len(val_loader) if val_loader is not None else None} "
            f"minibatch_size={_train_minibatch_size(config)} "
            f"accumulation_steps={_accumulation_steps(config)}"
        ),
        main_process=main_process,
        every_rank=True,
    )

    _debug_print(
        distributed_state,
        "building model",
        main_process=main_process,
        every_rank=True,
    )
    training_model = model or build_pmf_model(
        config.model,
        config.data,
        eval_mode=False,
    )
    training_model = _wrap_model(
        training_model,
        config,
        device=target_device,
        state=distributed_state,
    )
    _debug_print(
        distributed_state,
        "model ready",
        main_process=main_process,
        every_rank=True,
    )
    _debug_print(
        distributed_state,
        "building physics",
        main_process=main_process,
        every_rank=True,
    )
    training_physics = physics or build_physics(
        config.data,
        device=target_device,
        dtype=torch.float32,
    )
    _debug_print(
        distributed_state,
        "physics ready",
        main_process=main_process,
        every_rank=True,
    )
    training_optimizer = optimizer or _build_optimizer(config, training_model)
    training_scheduler = scheduler or _build_lr_scheduler(config, training_optimizer)
    training_scaler = scaler if scaler is not None else _build_scaler(config, target_device)
    if loss_fn is not None:
        training_loss = loss_fn
    elif config.training.loss_method == "imf":
        training_loss = IMFLoss(
            training_model,
            training_physics,
        )
    else:
        training_loss = OneStepLoss(
            training_model,
            training_physics,
            project_loss_to_measurement_space=config.training.project_loss_to_measurement_space,
            lpips=config.training.lpips,
            lpips_lambda=config.training.lpips_lambda,
            convnext=config.training.convnext,
            convnext_lambda=config.training.convnext_lambda,
            reconstruction_mse_lambda=config.training.reconstruction_mse_lambda,
            perceptual_max_t=config.training.perceptual_max_t,
            value_range=config.data.normalization_range,
            lpips_model=lpips_model,
            convnext_model=convnext_model,
        )
    _debug_print(
        distributed_state,
        "initializing wandb logger",
        main_process=main_process,
    )
    training_logger = logger or WandbLogger(
        config,
        is_main_process=main_process,
    )
    _debug_print(
        distributed_state,
        "wandb logger ready",
        main_process=main_process,
    )

    _debug_print(
        distributed_state,
        "checking checkpoint resume",
        main_process=main_process,
        every_rank=True,
    )
    resume_state = load_from_config(
        config,
        model=training_model,
        optimizer=training_optimizer,
        scaler=training_scaler,
        scheduler=training_scheduler,
        map_location=target_device,
    )
    if resume_state is not None and config.training.resume_learning_rate is not None:
        new_lrs = [config.training.resume_learning_rate] * len(training_optimizer.param_groups)
        _set_optimizer_lrs(training_optimizer, new_lrs)
    _set_runtime_seed(config, distributed_state)
    start_epoch = 1 if resume_state is None else resume_state.epoch + 1
    global_step = 0 if resume_state is None else resume_state.global_step
    _debug_print(
        distributed_state,
        f"resume ready start_epoch={start_epoch} global_step={global_step}",
        main_process=main_process,
        every_rank=True,
    )
    threshold_state = ValidationThresholdState.from_mapping(
        None
        if resume_state is None
        else dict(resume_state.engine_state.get("validation_threshold", {}))
    )
    completed_epochs = max(start_epoch - 1, 0)
    checkpoint_paths: list[Path] = []
    last_validation: ValidationResult | None = None

    try:
        for epoch in range(start_epoch, config.training.num_epochs + 1):
            _set_epoch(train_sampler, epoch)
            accumulation_steps = _accumulation_steps(config)
            step_interval = _debug_step_interval()
            epoch_step = 0
            pending_batches: list[Any] = []
            pending_data_time = 0.0
            previous_step_end = perf_counter()
            _debug_print(
                distributed_state,
                (
                    f"epoch {epoch}/{config.training.num_epochs} start "
                    f"local_train_batches={_maybe_len(train_loader)} "
                    f"global_step={global_step}"
                ),
                main_process=main_process,
                every_rank=True,
            )
            for batch in train_loader:
                pending_data_time += perf_counter() - previous_step_end
                pending_batches.append(batch)
                if len(pending_batches) < accumulation_steps:
                    previous_step_end = perf_counter()
                    continue

                step_start = perf_counter()
                step_result = run_train_step(
                    model=training_model,
                    optimizer=training_optimizer,
                    loss_fn=training_loss,
                    x=pending_batches,
                    training_config=config.training,
                    device=target_device,
                    scaler=training_scaler,
                )
                step_time = perf_counter() - step_start
                global_step += 1
                epoch_step += 1

                train_scalars = _build_train_scalars(
                    step_result,
                    data_time=pending_data_time,
                    step_time=step_time,
                )
                if global_step == 1 or global_step % step_interval == 0:
                    _debug_print(
                        distributed_state,
                        (
                            f"epoch {epoch} step {epoch_step} global_step={global_step} "
                            f"loss={train_scalars.get('train_loss', float('nan')):.6g} "
                            f"step_time={step_time:.3f}s"
                        ),
                        main_process=main_process,
                    )
                if main_process and _should_run(global_step, config.training.log_every_steps):
                    training_logger.log_scalars(train_scalars, global_step=global_step)

                pending_batches = []
                pending_data_time = 0.0
                previous_step_end = perf_counter()

            if pending_batches:
                step_start = perf_counter()
                step_result = run_train_step(
                    model=training_model,
                    optimizer=training_optimizer,
                    loss_fn=training_loss,
                    x=pending_batches,
                    training_config=config.training,
                    device=target_device,
                    scaler=training_scaler,
                )
                step_time = perf_counter() - step_start
                global_step += 1
                epoch_step += 1

                train_scalars = _build_train_scalars(
                    step_result,
                    data_time=pending_data_time,
                    step_time=step_time,
                )
                if global_step == 1 or global_step % step_interval == 0:
                    _debug_print(
                        distributed_state,
                        (
                            f"epoch {epoch} final step {epoch_step} global_step={global_step} "
                            f"loss={train_scalars.get('train_loss', float('nan')):.6g} "
                            f"step_time={step_time:.3f}s"
                        ),
                        main_process=main_process,
                    )
                if main_process and _should_run(global_step, config.training.log_every_steps):
                    training_logger.log_scalars(train_scalars, global_step=global_step)

            completed_epochs = epoch
            _debug_print(
                distributed_state,
                (
                    f"epoch {epoch} train loop done epoch_steps={epoch_step} "
                    f"global_step={global_step}"
                ),
                main_process=main_process,
                every_rank=True,
            )
            if training_scheduler is not None:
                training_scheduler.step()
            should_validate = _should_run(epoch, config.training.val_every_epochs)
            should_visualize = _should_run(epoch, config.training.vis_every_epochs)
            if main_process and (should_validate or should_visualize):
                assert val_loader is not None
                _debug_print(
                    distributed_state,
                    (
                        f"epoch {epoch} validation start "
                        f"val_batches={_maybe_len(val_loader)}"
                    ),
                    main_process=main_process,
                )
                last_validation = run_validation(
                    training_model,
                    training_physics,
                    val_loader,
                    config,
                    device=target_device,
                    lpips_model=lpips_model,
                    fid_extractor=fid_extractor,
                )
                _debug_print(
                    distributed_state,
                    (
                        f"epoch {epoch} validation done "
                        f"metrics={last_validation.scalars}"
                    ),
                    main_process=main_process,
                )
                if should_validate:
                    _debug_print(
                        distributed_state,
                        f"epoch {epoch} logging validation scalars",
                        main_process=main_process,
                    )
                    training_logger.log_scalars(
                        last_validation.scalars,
                        global_step=global_step,
                    )
                    threshold_scalars: dict[str, float] = {}
                    triggered_now = False
                    threshold_state, threshold_scalars, triggered_now = _apply_validation_threshold(
                        config,
                        training_optimizer,
                        last_validation.scalars,
                        threshold_state,
                        epoch=epoch,
                    )
                    if config.training.validation_threshold_enabled:
                        training_logger.log_scalars(
                            threshold_scalars,
                            global_step=global_step,
                        )
                if should_visualize and last_validation.visuals is not None:
                    _debug_print(
                        distributed_state,
                        f"epoch {epoch} logging validation grid",
                        main_process=main_process,
                    )
                    training_logger.log_grid(
                        ground_truth=last_validation.visuals.ground_truth,
                        measurement=last_validation.visuals.measurement,
                        null_e=last_validation.visuals.null_e,
                        reconstruction=last_validation.visuals.reconstruction,
                        global_step=global_step,
                    )
            elif should_validate:
                triggered_now = False
            if (
                config.training.validation_threshold_enabled
                and distributed_state.enabled
                and should_validate
            ):
                sync_path = _validation_threshold_sync_path(config.output_dir, epoch)
                _debug_print(
                    distributed_state,
                    f"epoch {epoch} validation-threshold sync start path={sync_path}",
                    main_process=main_process,
                    every_rank=True,
                )
                threshold_state = _sync_validation_threshold_state(
                    optimizer=training_optimizer,
                    threshold_state=threshold_state,
                    triggered_now=triggered_now,
                    distributed_state=distributed_state,
                    is_main_process=main_process,
                    sync_path=sync_path,
                )
                _debug_print(
                    distributed_state,
                    f"epoch {epoch} validation-threshold sync done",
                    main_process=main_process,
                    every_rank=True,
                )
            if main_process and epoch % config.training.checkpoint_every_epochs == 0:
                _debug_print(
                    distributed_state,
                    f"epoch {epoch} checkpoint save start",
                    main_process=main_process,
                )
                checkpoint_path = save_checkpoint(
                    _checkpoint_path(config, epoch),
                    model=training_model,
                    optimizer=training_optimizer,
                    scaler=training_scaler,
                    scheduler=training_scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    engine_state={
                        "validation_threshold": threshold_state.to_dict(),
                    },
                )
                checkpoint_paths.append(checkpoint_path)
                _debug_print(
                    distributed_state,
                    f"epoch {epoch} checkpoint saved path={checkpoint_path}",
                    main_process=main_process,
                )
    finally:
        _debug_print(
            distributed_state,
            f"finally start completed_epochs={completed_epochs} global_step={global_step}",
            main_process=main_process,
            every_rank=True,
        )
        if main_process:
            _debug_print(
                distributed_state,
                "wandb finish start",
                main_process=main_process,
            )
            training_logger.finish()
            _debug_print(
                distributed_state,
                "wandb finish done",
                main_process=main_process,
            )
        _debug_print(
            distributed_state,
            "distributed cleanup start",
            main_process=main_process,
            every_rank=True,
        )
        _cleanup_distributed(distributed_state)
        _debug_print(
            distributed_state,
            "distributed cleanup done",
            main_process=main_process,
            every_rank=True,
        )

    return EngineResult(
        start_epoch=start_epoch,
        completed_epochs=completed_epochs,
        global_step=global_step,
        checkpoint_paths=tuple(checkpoint_paths),
        last_validation=last_validation,
    )
