"""Configuration objects and YAML loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar


T = TypeVar("T")


@dataclass
class TrainingConfig:
    """Stores training-time hyperparameters and runtime intervals."""

    batch_size: int = 32
    minibatch_size: int | None = None
    num_epochs: int = 100
    loss_method: str = "pmf"
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "none"
    lr_scheduler_step_size: int = 1
    lr_scheduler_gamma: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip_norm: float | None = 1.0
    load_from: str | None = None
    resume_learning_rate: float | None = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    amp_enabled: bool = True
    distributed: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = True
    sync_batchnorm: bool = False
    project_loss_to_measurement_space: bool = True
    lpips: bool = False
    lpips_lambda: float = 1.0
    convnext: bool = False
    convnext_lambda: float = 0.0
    reconstruction_mse_lambda: float = 0.0
    perceptual_max_t: float = 1.0
    validation_threshold_enabled: bool = False
    validation_monitor_metric: str = "val_psnr"
    validation_metric_mode: str = "max"
    validation_threshold_value: float = 0.0
    validation_threshold_patience: int = 1
    validation_threshold_lr_scale: float = 0.1
    validation_threshold_reset_optimizer_state: bool = False
    log_every_steps: int = 100
    val_every_epochs: int = 1
    vis_every_epochs: int = 1
    checkpoint_every_epochs: int = 1

    def __post_init__(self) -> None:
        """Validates the training configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.minibatch_size is not None:
            if self.minibatch_size <= 0:
                raise ValueError("minibatch_size must be positive when provided.")
            if self.minibatch_size > self.batch_size:
                raise ValueError("minibatch_size must be less than or equal to batch_size.")
            if self.batch_size % self.minibatch_size != 0:
                raise ValueError("batch_size must be divisible by minibatch_size.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.loss_method not in {"pmf", "imf"}:
            raise ValueError("loss_method must be one of: pmf, imf.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.resume_learning_rate is not None and self.resume_learning_rate <= 0:
            raise ValueError("resume_learning_rate must be positive when provided.")
        if self.lr_scheduler_type not in {"none", "step"}:
            raise ValueError("lr_scheduler_type must be one of: none, step.")
        if self.lr_scheduler_step_size <= 0:
            raise ValueError("lr_scheduler_step_size must be positive.")
        if self.lr_scheduler_gamma <= 0:
            raise ValueError("lr_scheduler_gamma must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if not self.backend:
            raise ValueError("backend must be non-empty.")
        if self.backend not in {"nccl", "gloo", "mpi"}:
            raise ValueError("backend must be one of: nccl, gloo, mpi.")
        if self.lpips_lambda < 0:
            raise ValueError("lpips_lambda must be non-negative.")
        if self.convnext_lambda < 0:
            raise ValueError("convnext_lambda must be non-negative.")
        if self.reconstruction_mse_lambda < 0:
            raise ValueError("reconstruction_mse_lambda must be non-negative.")
        if self.perceptual_max_t <= 0 or self.perceptual_max_t > 1:
            raise ValueError("perceptual_max_t must lie in the interval (0, 1].")
        if not self.validation_monitor_metric:
            raise ValueError("validation_monitor_metric must be non-empty.")
        if self.validation_metric_mode not in {"max", "min"}:
            raise ValueError("validation_metric_mode must be one of: max, min.")
        if self.validation_threshold_patience <= 0:
            raise ValueError("validation_threshold_patience must be positive.")
        if self.validation_threshold_lr_scale <= 0:
            raise ValueError("validation_threshold_lr_scale must be positive.")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive.")
        if self.val_every_epochs <= 0:
            raise ValueError("val_every_epochs must be positive.")
        if self.vis_every_epochs <= 0:
            raise ValueError("vis_every_epochs must be positive.")
        if self.checkpoint_every_epochs <= 0:
            raise ValueError("checkpoint_every_epochs must be positive.")


@dataclass
class InpaintingDataConfig:
    """Stores inpainting-only data settings."""

    mask_type: str = "center"
    center_mask_size: int = 64
    fixed_mask_path: str | None = None
    fixed_mask_bank_dir: str | None = None
    fixed_mask_key: str | None = None
    precompute_a_dagger_ops: bool = False

    def __post_init__(self) -> None:
        """Validates the inpainting-only configuration."""
        if self.mask_type not in {"center", "fixed"}:
            raise ValueError("mask_type must be one of: center, fixed.")
        if self.center_mask_size <= 0:
            raise ValueError("center_mask_size must be positive.")
        if self.fixed_mask_path is not None and not self.fixed_mask_path:
            raise ValueError("fixed_mask_path must be non-empty when provided.")
        if self.fixed_mask_bank_dir is not None and not self.fixed_mask_bank_dir:
            raise ValueError("fixed_mask_bank_dir must be non-empty when provided.")
        if self.fixed_mask_key is not None and not self.fixed_mask_key:
            raise ValueError("fixed_mask_key must be non-empty when provided.")
        if self.mask_type == "fixed" and self.fixed_mask_path is None:
            if self.fixed_mask_bank_dir is None:
                raise ValueError(
                    "mask_type='fixed' requires one fixed mask path or bank directory."
                )


@dataclass
class DataConfig:
    """Stores dataset and preprocessing settings."""

    dataset_name: str = "ffhq"
    data_root: str = ""
    datasheet_path: str | None = None
    skip_missing_files: bool = True
    image_size: int = 256
    image_channels: int = 3
    train_split_start: int = 0
    train_split_end: int = 68_000
    val_split_start: int = 68_000
    val_split_end: int = 70_000
    test_split_start: int = 0
    test_split_end: int = 0
    enable_random_flip: bool = True
    normalization_range: tuple[float, float] = (-1.0, 1.0)
    resize_interpolation: str = "lanczos"
    cache_resized_images: bool = False
    resized_cache_dir: str | None = None
    inpainting: InpaintingDataConfig | Mapping[str, Any] | None = None
    # Legacy flat aliases kept for backwards compatibility.
    mask_type: str = "center"
    center_mask_size: int = 64
    fixed_mask_path: str | None = None
    fixed_mask_bank_dir: str | None = None
    fixed_mask_key: str | None = None
    precompute_a_dagger_ops: bool = False

    def __post_init__(self) -> None:
        """Validates the data configuration."""
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        if self.datasheet_path is not None and not self.datasheet_path:
            raise ValueError("datasheet_path must be non-empty when provided.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive.")
        if self.train_split_start < 0:
            raise ValueError("train_split_start must be non-negative.")
        if self.train_split_end <= self.train_split_start:
            raise ValueError("train_split_end must be greater than train_split_start.")
        def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
            return max(a_start, b_start) < min(a_end, b_end)
        has_val_split = self.val_split_end > self.val_split_start
        if has_val_split:
            if self.val_split_start < 0:
                raise ValueError("val_split_start must be non-negative.")
            if _overlaps(self.train_split_start, self.train_split_end,
                         self.val_split_start, self.val_split_end):
                raise ValueError("train and val splits must not overlap.")
        if self.test_split_end > self.test_split_start:
            if self.test_split_start < 0:
                raise ValueError("test_split_start must be non-negative.")
            if _overlaps(self.train_split_start, self.train_split_end,
                         self.test_split_start, self.test_split_end):
                raise ValueError("train and test splits must not overlap.")
            if has_val_split and _overlaps(self.val_split_start, self.val_split_end,
                         self.test_split_start, self.test_split_end):
                raise ValueError("val and test splits must not overlap.")
        value_min, value_max = self.normalization_range
        if value_min >= value_max:
            raise ValueError("normalization_range must be an increasing interval.")

        self.inpainting = self._coerce_inpainting_section(self.inpainting)
        self._sync_legacy_aliases()

        if self.dataset_name != "ffhq":
            raise ValueError("dataset_name must be ffhq.")
        if self.inpainting.mask_type == "center" and self.inpainting.center_mask_size > self.image_size:
            raise ValueError("center_mask_size must not exceed image_size.")

    def _coerce_inpainting_section(
        self,
        section: InpaintingDataConfig | Mapping[str, Any] | None,
    ) -> InpaintingDataConfig:
        """Builds one typed inpainting section from nested or legacy values."""
        if isinstance(section, InpaintingDataConfig):
            return section
        if isinstance(section, Mapping):
            return InpaintingDataConfig(**section)
        return InpaintingDataConfig(
            mask_type=self.mask_type,
            center_mask_size=self.center_mask_size,
            fixed_mask_path=self.fixed_mask_path,
            fixed_mask_bank_dir=self.fixed_mask_bank_dir,
            fixed_mask_key=self.fixed_mask_key,
            precompute_a_dagger_ops=self.precompute_a_dagger_ops,
        )

    def _sync_legacy_aliases(self) -> None:
        """Keeps existing flat attribute access working for the active task."""
        self.mask_type = self.inpainting.mask_type
        self.center_mask_size = self.inpainting.center_mask_size
        self.precompute_a_dagger_ops = self.inpainting.precompute_a_dagger_ops
        self.fixed_mask_path = self.inpainting.fixed_mask_path
        self.fixed_mask_bank_dir = self.inpainting.fixed_mask_bank_dir
        self.fixed_mask_key = self.inpainting.fixed_mask_key

    def to_dict(self) -> dict[str, Any]:
        """Returns a clean task-oriented dictionary representation."""
        payload: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "data_root": self.data_root,
            "datasheet_path": self.datasheet_path,
            "skip_missing_files": self.skip_missing_files,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
            "train_split_start": self.train_split_start,
            "train_split_end": self.train_split_end,
            "val_split_start": self.val_split_start,
            "val_split_end": self.val_split_end,
            "test_split_start": self.test_split_start,
            "test_split_end": self.test_split_end,
            "enable_random_flip": self.enable_random_flip,
            "normalization_range": self.normalization_range,
            "resize_interpolation": self.resize_interpolation,
            "cache_resized_images": self.cache_resized_images,
            "resized_cache_dir": self.resized_cache_dir,
        }
        payload["inpainting"] = asdict(self.inpainting)
        return payload


@dataclass
class ModelConfig:
    """Stores model preset selection."""

    model_name: str = "pmf_unet"
    conditional: bool = False
    num_classes: int = 1000
    unet_model_channels: int = 128
    unet_emb_channels: int | None = None
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: str = "32,16,8"
    unet_dropout: float = 0.0
    unet_channel_mult: str = ""
    unet_conv_resample: bool = True
    unet_use_checkpoint: bool = False
    unet_use_fp16: bool = False
    unet_num_heads: int = 1
    unet_num_head_channels: int = -1
    unet_num_heads_upsample: int = -1
    unet_use_scale_shift_norm: bool = True
    unet_resblock_updown: bool = True
    unet_use_new_attention_order: bool = False
    unet_use_aux_conditioning: bool = True
    unet_use_label_conditioning: bool = False

    def __post_init__(self) -> None:
        """Validates the model configuration."""
        if not self.model_name:
            raise ValueError("model_name must be non-empty.")
        if self.model_name != "pmf_unet":
            raise ValueError("model_name must be pmf_unet.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.unet_model_channels <= 0:
            raise ValueError("unet_model_channels must be positive.")
        if self.unet_emb_channels is not None and self.unet_emb_channels <= 0:
            raise ValueError("unet_emb_channels must be positive when provided.")
        if self.unet_num_res_blocks <= 0:
            raise ValueError("unet_num_res_blocks must be positive.")
        if self.unet_dropout < 0 or self.unet_dropout >= 1:
            raise ValueError("unet_dropout must lie in the interval [0, 1).")
        if self.unet_num_heads <= 0:
            raise ValueError("unet_num_heads must be positive.")
        if self.unet_num_head_channels == 0:
            raise ValueError("unet_num_head_channels must be -1 or positive.")
        if self.unet_num_head_channels < -1:
            raise ValueError("unet_num_head_channels must be -1 or positive.")
        if self.unet_num_heads_upsample == 0:
            raise ValueError("unet_num_heads_upsample must be -1 or positive.")
        if self.unet_num_heads_upsample < -1:
            raise ValueError("unet_num_heads_upsample must be -1 or positive.")


@dataclass
class EvaluationConfig:
    """Stores validation and one-step reconstruction settings."""

    metrics: tuple[str, ...] = ("psnr", "ssim", "lpips", "fid")
    val_batch_size: int = 8
    fid_real_stats_path: str | None = "fid_stats/ffhq256_val.npz"
    fixed_mask_seed: int | None = None
    fixed_noise_seed: int = 11
    visualization_subset_size: int = 3
    run_one_step_solver: bool = True
    sampling_steps: int = 1
    sampling_method: str = "meanflow_update"
    reconstruction_mode: str = "model_only"

    def __post_init__(self) -> None:
        """Validates the evaluation configuration."""
        if not self.metrics:
            raise ValueError("metrics must contain at least one metric name.")
        valid_metrics = {"psnr", "ssim", "lpips", "fid"}
        unknown_metrics = set(self.metrics) - valid_metrics
        if unknown_metrics:
            sorted_metrics = ", ".join(sorted(unknown_metrics))
            raise ValueError(f"Unknown evaluation metrics: {sorted_metrics}.")
        if self.val_batch_size <= 0:
            raise ValueError("val_batch_size must be positive.")
        if self.visualization_subset_size <= 0:
            raise ValueError("visualization_subset_size must be positive.")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive.")
        if self.sampling_method not in {"meanflow_update", "direct_prediction"}:
            raise ValueError(
                "sampling_method must be one of: meanflow_update, direct_prediction."
            )
        if self.reconstruction_mode not in {"model_only", "measurement_fill"}:
            raise ValueError(
                "reconstruction_mode must be one of: model_only, measurement_fill."
            )


@dataclass
class WandbConfig:
    """Stores Weights & Biases configuration."""

    enabled: bool = True
    project: str = "onestep-inpaint"
    entity: str | None = None
    group: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "online"
    log_images: bool = True

    def __post_init__(self) -> None:
        """Validates the Weights & Biases configuration."""
        if self.enabled and self.mode != "disabled" and not self.project:
            raise ValueError("project must be non-empty.")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("mode must be one of: online, offline, disabled.")


@dataclass
class ExperimentConfig:
    """Bundles the top-level experiment configuration objects."""

    experiment_name: str = "ffhq256_inpainting"
    task_name: str = "inpainting"
    seed: int = 42
    output_dir: str = "outputs"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        """Validates the experiment configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name must be non-empty.")
        if not self.task_name:
            raise ValueError("task_name must be non-empty.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """Returns a nested dictionary representation of the config."""
        return {
            "experiment_name": self.experiment_name,
            "task_name": self.task_name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "training": asdict(self.training),
            "data": self.data.to_dict(),
            "model": asdict(self.model),
            "evaluation": asdict(self.evaluation),
            "wandb": asdict(self.wandb),
        }

    def with_updates(self, **kwargs: Any) -> "ExperimentConfig":
        """Returns a shallowly updated copy of the config."""
        return replace(self, **kwargs)


def _validate_section_keys(
    section_name: str,
    values: Mapping[str, Any],
    section_type: type[Any],
) -> None:
    """Checks that a YAML section only contains known keys."""
    field_names = {item.name for item in fields(section_type)}
    unknown_keys = set(values) - field_names
    if unknown_keys:
        sorted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown keys in '{section_name}': {sorted_keys}.")


def _normalize_section_values(
    section_name: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalizes YAML lists to the tuple fields used in dataclasses."""
    normalized = dict(values)
    if section_name == "data" and "normalization_range" in normalized:
        normalized["normalization_range"] = tuple(normalized["normalization_range"])
    if section_name == "evaluation" and "metrics" in normalized:
        normalized["metrics"] = tuple(normalized["metrics"])
    if section_name == "wandb" and "tags" in normalized:
        normalized["tags"] = tuple(normalized["tags"])
    return normalized


def _build_section(
    section_name: str,
    section_type: type[T],
    values: Mapping[str, Any] | None,
) -> T:
    """Builds a typed config section from raw YAML data."""
    if values is None:
        return section_type()
    _validate_section_keys(section_name, values, section_type)
    normalized = _normalize_section_values(section_name, values)
    return section_type(**normalized)


def experiment_config_from_dict(config_dict: Mapping[str, Any]) -> ExperimentConfig:
    """Builds an experiment config from a nested dictionary."""
    top_level_keys = {
        "experiment_name",
        "task_name",
        "seed",
        "output_dir",
        "training",
        "data",
        "model",
        "evaluation",
        "wandb",
    }
    unknown_keys = set(config_dict) - top_level_keys
    if unknown_keys:
        sorted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown top-level config keys: {sorted_keys}.")

    top_level_values = {
        "experiment_name": config_dict.get("experiment_name", "ffhq256_inpainting"),
        "task_name": config_dict.get("task_name", "inpainting"),
        "seed": config_dict.get("seed", 42),
        "output_dir": config_dict.get("output_dir", "outputs"),
        "training": _build_section("training", TrainingConfig, config_dict.get("training")),
        "data": _build_section("data", DataConfig, config_dict.get("data")),
        "model": _build_section("model", ModelConfig, config_dict.get("model")),
        "evaluation": _build_section(
            "evaluation",
            EvaluationConfig,
            config_dict.get("evaluation"),
        ),
        "wandb": _build_section("wandb", WandbConfig, config_dict.get("wandb")),
    }
    return ExperimentConfig(**top_level_values)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Loads an experiment config from a YAML file."""
    import yaml

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}
    if not isinstance(config_dict, Mapping):
        raise ValueError("YAML config must decode to a mapping at the top level.")
    return experiment_config_from_dict(config_dict)


def save_experiment_config(
    config: ExperimentConfig,
    config_path: str | Path,
) -> None:
    """Saves an experiment config to a YAML file."""
    import yaml

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def get_default_experiment_config() -> ExperimentConfig:
    """Builds the default experiment configuration."""
    return ExperimentConfig()


__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "InpaintingDataConfig",
    "ModelConfig",
    "TrainingConfig",
    "WandbConfig",
    "experiment_config_from_dict",
    "get_default_experiment_config",
    "load_experiment_config",
    "save_experiment_config",
]
