"""Point-cloud metrics for offline oracle-gain labels."""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, object]:
        return {
            "chamfer": self.chamfer,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "fscore": {str(threshold): value for threshold, value in self.fscore.items()},
            "coverage": self.coverage,
        }


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


def sample_points(points: Tensor, max_points: int | None, seed: int = 0) -> Tensor:
    points = filter_finite_points(points)
    if max_points is None or points.shape[0] <= max_points:
        return points.cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(points.shape[0], generator=generator)[:max_points]
    return points.cpu()[indices]


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


def _estimate_similarity_transform(source: Tensor, target: Tensor) -> tuple[float, Tensor, Tensor]:
    """Estimate Sim(3) mapping source rows onto paired target rows."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("source and target correspondences must both have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("at least three correspondences are required for Sim(3)")

    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / source.shape[0]
    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = vh.T @ u.T
    if torch.linalg.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    variance = source_centered.square().sum() / source.shape[0]
    scale = (singular_values.sum() / variance.clamp_min(1e-12)).item()
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation, translation


def apply_similarity_transform(points: Tensor, scale: float, rotation: Tensor, translation: Tensor) -> Tensor:
    return scale * (points.float() @ rotation.T) + translation.float()


def align_sim3_icp(
    source: Tensor,
    target: Tensor,
    iterations: int = 6,
    sample_size: int = 4096,
    chunk_size: int = 1024,
    seed: int = 0,
) -> Tensor:
    """Align source to target with a deterministic point-to-point Sim(3) ICP."""

    source_full = filter_finite_points(source).float().cpu()
    target_sample = sample_points(target, sample_size, seed=seed + 1).float().cpu()
    source_sample = sample_points(source_full, sample_size, seed=seed).float().cpu()
    if source_sample.shape[0] < 3 or target_sample.shape[0] < 3:
        raise ValueError("source and target need at least three finite points for alignment")

    source_scale = source_sample.std(dim=0).norm().clamp_min(1e-12)
    target_scale = target_sample.std(dim=0).norm().clamp_min(1e-12)
    initial_scale = (target_scale / source_scale).item()
    initial_translation = target_sample.mean(dim=0) - initial_scale * source_sample.mean(dim=0)
    source_full = initial_scale * source_full + initial_translation
    source_sample = initial_scale * source_sample + initial_translation

    for _iteration in range(iterations):
        nearest = []
        for start in range(0, source_sample.shape[0], chunk_size):
            chunk = source_sample[start : start + chunk_size]
            distances = torch.cdist(chunk, target_sample)
            nearest.append(target_sample[distances.argmin(dim=1)])
        matched = torch.cat(nearest, dim=0)
        scale, rotation, translation = _estimate_similarity_transform(source_sample, matched)
        source_sample = apply_similarity_transform(source_sample, scale, rotation, translation)
        source_full = apply_similarity_transform(source_full, scale, rotation, translation)

    return source_full


def compute_pointcloud_metrics(
    reconstruction: Tensor,
    target: Tensor,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.1),
    coverage_radius: float = 0.05,
    max_points: int | None = 30000,
    voxel_size: float | None = None,
    chunk_size: int = 2048,
    seed: int = 0,
) -> PointCloudMetrics:
    reconstruction = voxel_downsample_points(reconstruction, voxel_size)
    target = voxel_downsample_points(target, voxel_size)
    reconstruction = sample_points(reconstruction, max_points, seed=seed).float().cpu()
    target = sample_points(target, max_points, seed=seed + 17).float().cpu()
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
