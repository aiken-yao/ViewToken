"""Point-cloud metrics for offline oracle-gain labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PointCloudMetrics:
    """Geometry quality metrics for an aligned reconstruction.

    Distance metrics are lower-is-better in raw form. Gain computation reverses
    them so every stored gain remains larger-is-better.
    """

    chamfer: float
    accuracy: float
    completeness: float
    fscore: dict[float, float]
    coverage: float
    reconstruction_count: int | None = None
    target_count: int | None = None
    voxel_size: float | None = None
    sample_method: str = "random"

    def to_dict(self) -> dict[str, object]:
        return {
            "chamfer": self.chamfer,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "fscore": {str(threshold): value for threshold, value in self.fscore.items()},
            "coverage": self.coverage,
            "reconstruction_count": self.reconstruction_count,
            "target_count": self.target_count,
            "voxel_size": self.voxel_size,
            "sample_method": self.sample_method,
        }


@dataclass(frozen=True)
class SimilarityTransform:
    """Row-vector Sim(3) transform: y = scale * (x @ R.T) + t."""

    scale: float
    rotation: Tensor
    translation: Tensor

    def apply(self, points: Tensor) -> Tensor:
        return apply_similarity_transform(
            points,
            scale=self.scale,
            rotation=self.rotation,
            translation=self.translation,
        )

    @property
    def rotation_angle_degrees(self) -> float:
        trace = torch.trace(self.rotation.float()).item()
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cosine))

    @property
    def translation_norm(self) -> float:
        return torch.linalg.norm(self.translation.float()).item()

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.float().cpu().tolist(),
            "translation": self.translation.float().cpu().tolist(),
            "rotation_angle_degrees": self.rotation_angle_degrees,
            "translation_norm": self.translation_norm,
        }


@dataclass(frozen=True)
class AlignmentDiagnostics:
    transform: SimilarityTransform
    iterations: int
    source_sample_count: int
    target_sample_count: int
    trim_fraction: float
    inlier_threshold: float | None
    residual_mean: float
    residual_median: float
    residual_rmse: float
    residual_max: float
    inlier_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform.to_dict(),
            "iterations": self.iterations,
            "source_sample_count": self.source_sample_count,
            "target_sample_count": self.target_sample_count,
            "trim_fraction": self.trim_fraction,
            "inlier_threshold": self.inlier_threshold,
            "residual_mean": self.residual_mean,
            "residual_median": self.residual_median,
            "residual_rmse": self.residual_rmse,
            "residual_max": self.residual_max,
            "inlier_ratio": self.inlier_ratio,
        }


@dataclass(frozen=True)
class PointCloudResidualDiagnostics:
    source_sample_count: int
    target_sample_count: int
    voxel_size: float | None
    max_points: int | None
    residual_mean: float
    residual_median: float
    residual_rmse: float
    residual_max: float
    inlier_ratios: dict[float, float]
    sample_method: str = "random"

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sample_count": self.source_sample_count,
            "target_sample_count": self.target_sample_count,
            "voxel_size": self.voxel_size,
            "max_points": self.max_points,
            "sample_method": self.sample_method,
            "residual_mean": self.residual_mean,
            "residual_median": self.residual_median,
            "residual_rmse": self.residual_rmse,
            "residual_max": self.residual_max,
            "inlier_ratios": {
                str(threshold): ratio
                for threshold, ratio in sorted(self.inlier_ratios.items())
            },
        }


@dataclass(frozen=True)
class AlignmentResult:
    points: Tensor
    diagnostics: AlignmentDiagnostics


def filter_finite_points(points: Tensor) -> Tensor:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    return points[torch.isfinite(points).all(dim=-1)]


def voxel_downsample_points(points: Tensor, voxel_size: float | None) -> Tensor:
    """Average points that fall in the same voxel.

    The caller is responsible for applying this after any alignment if the voxel
    size is meant to be interpreted in ground-truth metric units.
    """

    points = filter_finite_points(points).float().cpu()
    if voxel_size is None or voxel_size <= 0 or points.numel() == 0:
        return points

    voxel_indices = torch.floor(points / float(voxel_size)).to(torch.int64)
    unique_voxels, inverse = torch.unique(
        voxel_indices, dim=0, sorted=True, return_inverse=True
    )
    sums = torch.zeros((unique_voxels.shape[0], 3), dtype=points.dtype)
    sums.index_add_(0, inverse.cpu(), points)
    counts = torch.bincount(inverse.cpu(), minlength=unique_voxels.shape[0]).to(
        points.dtype
    )
    return sums / counts.clamp_min(1).unsqueeze(-1)


def sample_point_indices(
    points: Tensor,
    max_points: int | None,
    seed: int = 0,
    method: str = "random",
) -> Tensor:
    """Return row indices for either random or deterministic point sampling."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    method = method.lower()
    if method in {"none", "all"} or max_points is None or max_points <= 0:
        return torch.arange(points.shape[0], dtype=torch.long)
    if points.shape[0] <= max_points:
        return torch.arange(points.shape[0], dtype=torch.long)
    if method == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randperm(points.shape[0], generator=generator)[:max_points]
    if method == "hash":
        sanitized = torch.nan_to_num(points.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        quantized = torch.round(sanitized * 10000.0).to(torch.int64)
        hash_values = torch.remainder(
            quantized[:, 0] * 73856093
            + quantized[:, 1] * 19349663
            + quantized[:, 2] * 83492791
            + int(seed) * 2654435761,
            2147483647,
        )
        return torch.argsort(hash_values, stable=True)[:max_points].cpu()
    raise ValueError(f"Unsupported point sampling method: {method}")


def sample_points(
    points: Tensor,
    max_points: int | None,
    seed: int = 0,
    method: str = "random",
) -> Tensor:
    points = filter_finite_points(points).float().cpu()
    indices = sample_point_indices(points, max_points=max_points, seed=seed, method=method)
    return points[indices]


def nearest_neighbor_squared_distances(
    source: Tensor,
    target: Tensor,
    chunk_size: int = 2048,
) -> Tensor:
    """Return squared nearest-neighbor distances from source to target."""

    source = filter_finite_points(source).float().cpu()
    target = filter_finite_points(target).float().cpu()
    if source.numel() == 0 or target.numel() == 0:
        raise ValueError("source and target point clouds must be non-empty")

    distances = []
    target = target.contiguous()
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size].contiguous()
        squared = torch.cdist(chunk, target).square()
        distances.append(squared.min(dim=1).values.cpu())
    return torch.cat(distances, dim=0)


def nearest_neighbor_distances_and_indices(
    source: Tensor,
    target: Tensor,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Return nearest-neighbor Euclidean distances and target indices."""

    source = filter_finite_points(source).float().cpu()
    target = filter_finite_points(target).float().cpu()
    if source.numel() == 0 or target.numel() == 0:
        raise ValueError("source and target point clouds must be non-empty")

    distances = []
    indices = []
    target = target.contiguous()
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size].contiguous()
        chunk_distances = torch.cdist(chunk, target)
        values, nearest = chunk_distances.min(dim=1)
        distances.append(values.cpu())
        indices.append(nearest.cpu())
    return torch.cat(distances, dim=0), torch.cat(indices, dim=0)


def estimate_similarity_transform(source: Tensor, target: Tensor) -> SimilarityTransform:
    """Estimate Sim(3) mapping paired source rows onto paired target rows."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("source and target correspondences must both have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("at least three correspondences are required for Sim(3)")

    source = source.float().cpu()
    target = target.float().cpu()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / source.shape[0]
    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = vh.T @ u.T
    det_correction = 1.0
    if torch.linalg.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
        det_correction = -1.0
    signed_singular_values = singular_values.clone()
    signed_singular_values[-1] *= det_correction
    variance = source_centered.square().sum() / source.shape[0]
    scale = (signed_singular_values.sum() / variance.clamp_min(1e-12)).item()
    translation = target_mean - scale * (source_mean @ rotation.T)
    return SimilarityTransform(
        scale=scale,
        rotation=rotation.contiguous(),
        translation=translation.contiguous(),
    )


def compose_similarity_transforms(
    first: SimilarityTransform, second: SimilarityTransform
) -> SimilarityTransform:
    """Return a transform equivalent to applying first, then second."""

    scale = second.scale * first.scale
    rotation = second.rotation.float() @ first.rotation.float()
    translation = second.scale * (first.translation.float() @ second.rotation.float().T) + second.translation.float()
    return SimilarityTransform(
        scale=scale,
        rotation=rotation.contiguous(),
        translation=translation.contiguous(),
    )


def apply_similarity_transform(points: Tensor, scale: float, rotation: Tensor, translation: Tensor) -> Tensor:
    return scale * (points.float().cpu() @ rotation.float().cpu().T) + translation.float().cpu()


def _initial_similarity_by_scale_and_centroid(source: Tensor, target: Tensor) -> SimilarityTransform:
    source_scale = source.std(dim=0).norm().clamp_min(1e-12)
    target_scale = target.std(dim=0).norm().clamp_min(1e-12)
    scale = (target_scale / source_scale).item()
    translation = target.mean(dim=0) - scale * source.mean(dim=0)
    return SimilarityTransform(
        scale=scale,
        rotation=torch.eye(3, dtype=torch.float32),
        translation=translation.contiguous(),
    )


def _trim_correspondences(
    source: Tensor,
    target: Tensor,
    distances: Tensor,
    trim_fraction: float,
) -> tuple[Tensor, Tensor]:
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0, 1]")
    if trim_fraction == 1:
        return source, target
    keep_count = max(3, int(round(source.shape[0] * trim_fraction)))
    keep_count = min(keep_count, source.shape[0])
    keep = torch.topk(distances, k=keep_count, largest=False).indices
    return source[keep], target[keep]


def _alignment_residuals(source: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    distances, _indices = nearest_neighbor_distances_and_indices(
        source, target, chunk_size=chunk_size
    )
    return distances.float().cpu()


def _make_alignment_diagnostics(
    transform: SimilarityTransform,
    residuals: Tensor,
    iterations: int,
    source_sample_count: int,
    target_sample_count: int,
    trim_fraction: float,
    inlier_threshold: float | None,
) -> AlignmentDiagnostics:
    inlier_ratio = None
    if inlier_threshold is not None:
        inlier_ratio = (residuals <= float(inlier_threshold)).float().mean().item()
    return AlignmentDiagnostics(
        transform=transform,
        iterations=iterations,
        source_sample_count=source_sample_count,
        target_sample_count=target_sample_count,
        trim_fraction=trim_fraction,
        inlier_threshold=inlier_threshold,
        residual_mean=residuals.mean().item(),
        residual_median=residuals.median().item(),
        residual_rmse=residuals.square().mean().sqrt().item(),
        residual_max=residuals.max().item(),
        inlier_ratio=inlier_ratio,
    )


def compute_pointcloud_residual_diagnostics(
    source: Tensor,
    target: Tensor,
    inlier_thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    max_points: int | None = 12000,
    voxel_size: float | None = None,
    chunk_size: int = 2048,
    seed: int = 0,
    sample_method: str = "random",
) -> PointCloudResidualDiagnostics:
    """Measure source-to-target nearest-neighbor residuals after alignment."""

    source = voxel_downsample_points(source, voxel_size)
    target = voxel_downsample_points(target, voxel_size)
    source = sample_points(source, max_points, seed=seed, method=sample_method).float().cpu()
    target = sample_points(target, max_points, seed=seed + 17, method=sample_method).float().cpu()
    residuals = nearest_neighbor_squared_distances(
        source, target, chunk_size=chunk_size
    ).sqrt()
    return PointCloudResidualDiagnostics(
        source_sample_count=int(source.shape[0]),
        target_sample_count=int(target.shape[0]),
        voxel_size=voxel_size,
        max_points=max_points,
        sample_method=sample_method,
        residual_mean=residuals.mean().item(),
        residual_median=residuals.median().item(),
        residual_rmse=residuals.square().mean().sqrt().item(),
        residual_max=residuals.max().item(),
        inlier_ratios={
            float(threshold): (residuals <= float(threshold)).float().mean().item()
            for threshold in inlier_thresholds
        },
    )


def align_sim3_icp_with_diagnostics(
    source: Tensor,
    target: Tensor,
    iterations: int = 6,
    sample_size: int = 4096,
    chunk_size: int = 1024,
    seed: int = 0,
    trim_fraction: float = 1.0,
    inlier_threshold: float | None = None,
) -> AlignmentResult:
    """Align source to target with deterministic point-to-point Sim(3) ICP."""

    source_full = filter_finite_points(source).float().cpu()
    target_sample = sample_points(target, sample_size, seed=seed + 1).float().cpu()
    source_sample = sample_points(source_full, sample_size, seed=seed).float().cpu()
    if source_sample.shape[0] < 3 or target_sample.shape[0] < 3:
        raise ValueError("source and target need at least three finite points for alignment")

    total_transform = _initial_similarity_by_scale_and_centroid(source_sample, target_sample)
    source_full = total_transform.apply(source_full)
    source_sample = total_transform.apply(source_sample)

    for _iteration in range(iterations):
        distances, nearest_indices = nearest_neighbor_distances_and_indices(
            source_sample, target_sample, chunk_size=chunk_size
        )
        matched = target_sample[nearest_indices]
        fit_source, fit_target = _trim_correspondences(
            source_sample, matched, distances, trim_fraction=trim_fraction
        )
        step_transform = estimate_similarity_transform(fit_source, fit_target)
        source_sample = step_transform.apply(source_sample)
        source_full = step_transform.apply(source_full)
        total_transform = compose_similarity_transforms(total_transform, step_transform)

    residuals = _alignment_residuals(source_sample, target_sample, chunk_size=chunk_size)
    diagnostics = _make_alignment_diagnostics(
        transform=total_transform,
        residuals=residuals,
        iterations=iterations,
        source_sample_count=int(source_sample.shape[0]),
        target_sample_count=int(target_sample.shape[0]),
        trim_fraction=trim_fraction,
        inlier_threshold=inlier_threshold,
    )
    return AlignmentResult(points=source_full.contiguous(), diagnostics=diagnostics)


def align_sim3_icp(
    source: Tensor,
    target: Tensor,
    iterations: int = 6,
    sample_size: int = 4096,
    chunk_size: int = 1024,
    seed: int = 0,
) -> Tensor:
    """Align source to target with deterministic point-to-point Sim(3) ICP."""

    return align_sim3_icp_with_diagnostics(
        source,
        target,
        iterations=iterations,
        sample_size=sample_size,
        chunk_size=chunk_size,
        seed=seed,
    ).points


def compute_pointcloud_metrics(
    reconstruction: Tensor,
    target: Tensor,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    coverage_radius: float = 0.05,
    max_points: int | None = 30000,
    voxel_size: float | None = None,
    chunk_size: int = 2048,
    seed: int = 0,
    sample_method: str = "random",
) -> PointCloudMetrics:
    reconstruction = voxel_downsample_points(reconstruction, voxel_size)
    target = voxel_downsample_points(target, voxel_size)
    reconstruction = sample_points(
        reconstruction, max_points, seed=seed, method=sample_method
    ).float().cpu()
    target = sample_points(
        target, max_points, seed=seed + 17, method=sample_method
    ).float().cpu()
    source_to_target = nearest_neighbor_squared_distances(
        reconstruction, target, chunk_size=chunk_size
    ).sqrt()
    target_to_source = nearest_neighbor_squared_distances(
        target, reconstruction, chunk_size=chunk_size
    ).sqrt()

    accuracy = source_to_target.mean().item()
    completeness = target_to_source.mean().item()
    fscore = {}
    for threshold in thresholds:
        precision = (source_to_target <= threshold).float().mean().item()
        recall = (target_to_source <= threshold).float().mean().item()
        denominator = precision + recall
        fscore[threshold] = 0.0 if denominator == 0 else 2 * precision * recall / denominator

    coverage = (target_to_source <= coverage_radius).float().mean().item()
    chamfer = 0.5 * (accuracy + completeness)
    return PointCloudMetrics(
        chamfer=chamfer,
        accuracy=accuracy,
        completeness=completeness,
        fscore=fscore,
        coverage=coverage,
        reconstruction_count=int(reconstruction.shape[0]),
        target_count=int(target.shape[0]),
        voxel_size=voxel_size,
        sample_method=sample_method,
    )


def compute_metric_gains(
    baseline: PointCloudMetrics, candidate: PointCloudMetrics
) -> dict[str, object]:
    """Return gains where larger is better for every value."""

    thresholds = sorted(set(baseline.fscore) | set(candidate.fscore))
    return {
        "chamfer": baseline.chamfer - candidate.chamfer,
        "accuracy": baseline.accuracy - candidate.accuracy,
        "completeness": baseline.completeness - candidate.completeness,
        "fscore": {
            str(threshold): candidate.fscore.get(threshold, 0.0)
            - baseline.fscore.get(threshold, 0.0)
            for threshold in thresholds
        },
        "coverage": candidate.coverage - baseline.coverage,
    }
