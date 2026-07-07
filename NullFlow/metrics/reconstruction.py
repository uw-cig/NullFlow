"""Reconstruction metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


MetricName = str
_VALID_METRICS = {"psnr", "ssim", "lpips", "fid"}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _check_images(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Checks a prediction-target image pair before metric computation."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape (B, C, H, W).")
    if not torch.is_floating_point(prediction) or not torch.is_floating_point(target):
        raise ValueError("prediction and target must be floating-point tensors.")


def _resolve_path(path_value: str | Path) -> Path:
    """Resolves a path relative to the repository root when needed."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _to_unit_interval(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Maps images from the configured range to ``[0, 1]``."""
    value_min, value_max = value_range
    scale = value_max - value_min
    if scale <= 0:
        raise ValueError("value_range must be an increasing interval.")
    return ((images - value_min) / scale).clamp(0.0, 1.0)


def _to_uint8(
    images: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Maps floating-point images to uint8 tensors for FID features."""
    unit_images = _to_unit_interval(images, value_range)
    return (unit_images * 255.0).round().to(torch.uint8)


def _is_complex_channel_image(images: torch.Tensor) -> bool:
    """Checks whether one batch uses the repository's real/imag MRI encoding."""
    return images.ndim == 4 and images.shape[1] == 2


def _complex_magnitude(images: torch.Tensor) -> torch.Tensor:
    """Converts ``(B, 2, H, W)`` real/imag images to magnitude images."""
    if not _is_complex_channel_image(images):
        raise ValueError("MRI complex-channel images must have shape (B, 2, H, W).")
    return torch.sqrt(torch.sum(images.square(), dim=1, keepdim=True).clamp_min(0.0))


def _crop_border(images: torch.Tensor, border: int) -> torch.Tensor:
    """Crops one fixed image border when the spatial size is large enough."""
    if border <= 0:
        return images
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")
    height, width = images.shape[-2:]
    cropped_height = height - (2 * border)
    cropped_width = width - (2 * border)
    if cropped_height < 7 or cropped_width < 7:
        return images
    return images[:, :, border : height - border, border : width - border]


def _prepare_psnr_images(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the image tensors used by the PSNR implementation."""
    _check_images(prediction, target)
    if _is_complex_channel_image(prediction):
        pred_metric = _crop_border(_complex_magnitude(prediction), border=2)
        target_metric = _crop_border(_complex_magnitude(target), border=2)
        return pred_metric, target_metric
    return _to_unit_interval(prediction, value_range), _to_unit_interval(target, value_range)


def _prepare_ssim_images(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the image tensors used by the SSIM implementation."""
    _check_images(prediction, target)
    if _is_complex_channel_image(prediction):
        pred_metric = _crop_border(_complex_magnitude(prediction), border=4)
        target_metric = _crop_border(_complex_magnitude(target), border=4)
        return pred_metric, target_metric
    return _to_unit_interval(prediction, value_range), _to_unit_interval(target, value_range)


def _prepare_lpips_images(
    images: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Returns LPIPS-ready images in ``[-1, 1]`` and ``3`` channels."""
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")

    if _is_complex_channel_image(images):
        magnitude = _complex_magnitude(images).clamp(0.0, 1.0)
        return magnitude.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)

    if images.shape[1] == 1:
        unit_images = _to_unit_interval(images, value_range)
        return unit_images.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)

    if images.shape[1] == 3:
        return _to_unit_interval(images, value_range).mul(2.0).sub(1.0)

    raise ValueError("LPIPS expects images with 1, 2, or 3 channels.")


def _prepare_fid_images(
    images: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Returns FID-ready image batches with three channels in ``[0, 1]``."""
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")

    if _is_complex_channel_image(images):
        magnitude = _complex_magnitude(images).clamp(0.0, 1.0)
        return magnitude.repeat(1, 3, 1, 1)

    if images.shape[1] == 1:
        unit_images = _to_unit_interval(images, value_range)
        return unit_images.repeat(1, 3, 1, 1)

    if images.shape[1] == 3:
        return _to_unit_interval(images, value_range)

    raise ValueError("FID expects images with 1, 2, or 3 channels.")


def _psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Computes PSNR for each image in a batch."""
    pred_metric, target_metric = _prepare_psnr_images(
        prediction,
        target,
        value_range=value_range,
    )
    mse = torch.mean((pred_metric - target_metric) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse)


def _ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Computes SSIM for each image in a batch."""
    pred_metric, target_metric = _prepare_ssim_images(
        prediction,
        target,
        value_range=value_range,
    )
    pred_unit = pred_metric.detach().cpu()
    target_unit = target_metric.detach().cpu()
    scores = []
    for pred_image, target_image in zip(pred_unit, target_unit, strict=True):
        if pred_image.shape[0] == 1:
            score = structural_similarity(
                pred_image[0].numpy(),
                target_image[0].numpy(),
                data_range=1.0,
                win_size=7,
            )
        else:
            pred_hwc = pred_image.permute(1, 2, 0).numpy()
            target_hwc = target_image.permute(1, 2, 0).numpy()
            score = structural_similarity(
                pred_hwc,
                target_hwc,
                channel_axis=-1,
                data_range=1.0,
            )
        scores.append(score)
    return torch.tensor(scores, dtype=prediction.dtype)


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Computes the mean PSNR across a batch."""
    return float(_psnr_per_image(prediction, target, value_range=value_range).mean().cpu())


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Computes the mean SSIM across a batch."""
    return float(_ssim_per_image(prediction, target, value_range=value_range).mean().cpu())


class LPIPSMetric:
    """Computes LPIPS for image batches."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        net: str = "alex",
        model: torch.nn.Module | None = None,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initializes the LPIPS metric wrapper."""
        self.device = torch.device(device)
        self.net = net
        self._model = model
        self.value_range = value_range

    def _get_model(self) -> torch.nn.Module:
        """Returns the cached LPIPS model, creating it on first use."""
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net=self.net).to(self.device)
            self._model.eval()
        return self._model

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes LPIPS for each image in a batch."""
        _check_images(prediction, target)
        model = self._get_model()
        pred_metric = _prepare_lpips_images(prediction, value_range=self.value_range)
        target_metric = _prepare_lpips_images(target, value_range=self.value_range)
        with torch.no_grad():
            scores = model(
                pred_metric.to(self.device),
                target_metric.to(self.device),
            )
        return scores.reshape(-1).detach().cpu()


@dataclass
class _FeatureStats:
    """Stores sufficient statistics for one FID distribution."""

    count: int = 0
    sum: torch.Tensor | None = None
    sum_outer: torch.Tensor | None = None

    def update(self, features: torch.Tensor) -> None:
        """Updates the running statistics from a batch of features."""
        if features.ndim != 2:
            raise ValueError("features must have shape (B, D).")
        batch_features = features.to(dtype=torch.float64, device="cpu")
        batch_sum = batch_features.sum(dim=0)
        batch_outer = batch_features.T @ batch_features

        if self.sum is None:
            self.sum = batch_sum
            self.sum_outer = batch_outer
        else:
            self.sum = self.sum + batch_sum
            assert self.sum_outer is not None
            self.sum_outer = self.sum_outer + batch_outer
        self.count += int(batch_features.shape[0])

    def to_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Converts the running statistics to mean and covariance."""
        if self.count < 2:
            raise ValueError("FID requires at least two samples per distribution.")
        assert self.sum is not None
        assert self.sum_outer is not None
        mu = self.sum / self.count
        cov = (self.sum_outer - self.count * torch.outer(mu, mu)) / (self.count - 1)
        return mu.numpy(), cov.numpy()


def save_fid_stats(
    stats_path: str | Path,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    count: int,
) -> Path:
    """Saves FID reference statistics to a ``.npz`` file."""
    resolved_path = _resolve_path(stats_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        resolved_path,
        mu=np.asarray(mu, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        count=np.asarray(count, dtype=np.int64),
    )
    return resolved_path


def load_fid_stats(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Loads cached FID reference statistics from a ``.npz`` file."""
    resolved_path = _resolve_path(stats_path)
    payload = np.load(resolved_path)
    if "mu" not in payload or "sigma" not in payload or "count" not in payload:
        raise ValueError("FID stats file must contain 'mu', 'sigma', and 'count'.")
    mu = np.asarray(payload["mu"], dtype=np.float64)
    sigma = np.asarray(payload["sigma"], dtype=np.float64)
    count = int(payload["count"])
    return mu, sigma, count


class FIDMetric:
    """Accumulates batch updates and computes FID at the end."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        value_range: tuple[float, float] = (-1.0, 1.0),
        real_stats_path: str | Path | None = None,
        extractor: torch.nn.Module | None = None,
    ) -> None:
        """Initializes the FID accumulator."""
        self.device = torch.device(device)
        self.value_range = value_range
        self.real_stats_path = real_stats_path
        self._extractor = extractor
        self._real_stats: tuple[np.ndarray, np.ndarray, int] | None = None
        if real_stats_path is not None:
            self._real_stats = load_fid_stats(real_stats_path)
        self.reset()

    def reset(self) -> None:
        """Clears the accumulated real and fake feature statistics."""
        self._fake = _FeatureStats()
        if self._real_stats is None:
            self._real = _FeatureStats()
        else:
            self._real = None

    def _get_extractor(self) -> torch.nn.Module:
        """Returns the cached feature extractor, creating it on first use."""
        if self._extractor is None:
            from torch_fidelity.utils import create_feature_extractor

            self._extractor = create_feature_extractor(
                "inception-v3-compat",
                ["2048"],
                cuda=self.device.type == "cuda",
                verbose=False,
            )
        return self._extractor

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts FID features for one image batch."""
        _check_images(images, images)
        extractor = self._get_extractor()
        batch = (
            _prepare_fid_images(images, value_range=self.value_range)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .to(self.device)
        )
        with torch.no_grad():
            features = extractor(batch)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> None:
        """Updates the running FID statistics for one batch."""
        if target is not None:
            _check_images(prediction, target)
        elif prediction.ndim != 4:
            raise ValueError("prediction must have shape (B, C, H, W).")
        self._fake.update(self._extract_features(prediction))
        if self._real_stats is None:
            if target is None:
                raise ValueError("target is required when no cached real FID stats are provided.")
            assert self._real is not None
            self._real.update(self._extract_features(target))

    def update_real(
        self,
        target: torch.Tensor,
    ) -> None:
        """Updates only the real-image statistics for cached FID references."""
        if self._real_stats is not None:
            raise ValueError("Cannot update cached real stats after loading a .npz reference.")
        _check_images(target, target)
        assert self._real is not None
        self._real.update(self._extract_features(target))

    def compute(self) -> float:
        """Computes FID from the accumulated real and fake features."""
        if self._real_stats is None:
            assert self._real is not None
            mu_real, cov_real = self._real.to_statistics()
        else:
            mu_real, cov_real, _ = self._real_stats
        mu_fake, cov_fake = self._fake.to_statistics()
        diff = mu_real - mu_fake
        cov_prod = cov_real.dot(cov_fake)
        tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(cov_prod).astype("complex128")).real)
        fid = float(diff.dot(diff) + np.trace(cov_real) + np.trace(cov_fake) - 2.0 * tr_covmean)
        return max(fid, 0.0)

    def save_real_stats(self, stats_path: str | Path) -> Path:
        """Saves the accumulated real feature statistics to a ``.npz`` file."""
        if self._real_stats is not None:
            mu, sigma, count = self._real_stats
        else:
            assert self._real is not None
            mu, sigma = self._real.to_statistics()
            count = self._real.count
        return save_fid_stats(
            stats_path,
            mu=mu,
            sigma=sigma,
            count=count,
        )


def build_fid_stats(
    dataset: Dataset,
    stats_path: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | str = "cpu",
    value_range: tuple[float, float] = (-1.0, 1.0),
    extractor: torch.nn.Module | None = None,
    show_progress: bool = True,
) -> Path:
    """Builds and saves cached FID reference stats from one image dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    metric = FIDMetric(
        device=device,
        value_range=value_range,
        extractor=extractor,
    )
    progress = tqdm(
        total=len(dataset),
        desc="Building FID stats",
        unit="images",
        disable=not show_progress,
    )
    try:
        for batch in loader:
            batch_images: torch.Tensor
            if torch.is_tensor(batch):
                batch_images = batch
            elif isinstance(batch, Mapping) and torch.is_tensor(batch.get("x")):
                batch_images = batch["x"]
            else:
                raise ValueError("FID reference datasets must return a tensor batch or a dict with an 'x' tensor.")
            metric.update_real(batch_images)
            progress.update(int(batch_images.shape[0]))
    finally:
        progress.close()
    return metric.save_real_stats(stats_path)


class ReconstructionMetrics:
    """Tracks reconstruction metrics across validation batches."""

    def __init__(
        self,
        metrics: Sequence[MetricName],
        *,
        device: torch.device | str = "cpu",
        value_range: tuple[float, float] = (-1.0, 1.0),
        fid_real_stats_path: str | Path | None = None,
        lpips_model: torch.nn.Module | None = None,
        fid_extractor: torch.nn.Module | None = None,
    ) -> None:
        """Initializes the configured metric suite."""
        unknown_metrics = set(metrics) - _VALID_METRICS
        if unknown_metrics:
            names = ", ".join(sorted(unknown_metrics))
            raise ValueError(f"Unknown metrics: {names}.")

        self.metrics = tuple(metrics)
        self.device = torch.device(device)
        self.value_range = value_range
        self._totals = {name: 0.0 for name in self.metrics if name != "fid"}
        self._count = 0
        self._lpips = (
            LPIPSMetric(
                device=self.device,
                model=lpips_model,
                value_range=self.value_range,
            )
            if "lpips" in self.metrics
            else None
        )
        self._fid = (
            FIDMetric(
                device=self.device,
                value_range=self.value_range,
                real_stats_path=fid_real_stats_path,
                extractor=fid_extractor,
            )
            if "fid" in self.metrics
            else None
        )

    def reset(self) -> None:
        """Clears the accumulated metric state."""
        self._totals = {name: 0.0 for name in self.metrics if name != "fid"}
        self._count = 0
        if self._fid is not None:
            self._fid.reset()

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Updates the tracked metrics from one batch and returns batch means."""
        _check_images(prediction, target)
        batch_size = prediction.shape[0]
        batch_metrics: dict[str, float] = {}

        if "psnr" in self.metrics:
            psnr_scores = _psnr_per_image(prediction, target, value_range=self.value_range)
            batch_metrics["psnr"] = float(psnr_scores.mean().cpu())
            self._totals["psnr"] += float(psnr_scores.sum().cpu())

        if "ssim" in self.metrics:
            ssim_scores = _ssim_per_image(prediction, target, value_range=self.value_range)
            batch_metrics["ssim"] = float(ssim_scores.mean().cpu())
            self._totals["ssim"] += float(ssim_scores.sum().cpu())

        if self._lpips is not None:
            lpips_scores = self._lpips(prediction, target)
            batch_metrics["lpips"] = float(lpips_scores.mean().cpu())
            self._totals["lpips"] += float(lpips_scores.sum().cpu())

        if self._fid is not None:
            self._fid.update(prediction, target)

        self._count += batch_size
        return batch_metrics

    def compute(self) -> dict[str, float]:
        """Computes the aggregated metrics collected so far."""
        if self._count == 0:
            raise ValueError("No samples have been accumulated.")

        out = {
            name: total / self._count
            for name, total in self._totals.items()
        }
        if self._fid is not None:
            out["fid"] = self._fid.compute()
        return out
