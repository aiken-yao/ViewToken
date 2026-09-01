"""Stage-A calibration helpers for oracle metric and alignment noise."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import torch
from torch import Tensor

from .metrics import (
    AlignmentDiagnostics,
    PointCloudMetrics,
    align_sim3_icp_with_diagnostics,
    compute_metric_gains,
    compute_pointcloud_metrics,
)


@dataclass(frozen=True)
class AlignmentMetricResult:
    seed: int
    metrics: PointCloudMetrics
    alignment: AlignmentDiagnostics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "alignment": None if self.alignment is None else self.alignment.to_dict(),
        }


def flatten_metrics(metrics: PointCloudMetrics) -> dict[str, float]:
    flat = {
        "chamfer": metrics.chamfer,
        "accuracy": metrics.accuracy,
        "completeness": metrics.completeness,
        "coverage": metrics.coverage,
    }
    for threshold, value in metrics.fscore.items():
        flat[f"fscore@{threshold}"] = value
    return flat


def flatten_gains(gains: dict[str, object]) -> dict[str, float]:
    flat = {
        "chamfer": float(gains["chamfer"]),
        "accuracy": float(gains["accuracy"]),
        "completeness": float(gains["completeness"]),
        "coverage": float(gains["coverage"]),
    }
    for threshold, value in gains["fscore"].items():
        flat[f"fscore@{threshold}"] = float(value)
    return flat


def evaluate_alignment_and_metrics(
    points: Tensor,
    target: Tensor,
    alignment: str,
    seed: int,
    thresholds: tuple[float, ...],
    coverage_radius: float,
    max_points: int | None,
    voxel_size: float | None,
    trim_fraction: float = 1.0,
    inlier_threshold: float | None = None,
) -> AlignmentMetricResult:
    if alignment == "identity":
        aligned_points = points.float().cpu()
        diagnostics = None
    elif alignment == "sim3_icp":
        alignment_result = align_sim3_icp_with_diagnostics(
            points,
            target,
            seed=seed,
            trim_fraction=trim_fraction,
            inlier_threshold=inlier_threshold,
        )
        aligned_points = alignment_result.points
        diagnostics = alignment_result.diagnostics
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")

    metrics = compute_pointcloud_metrics(
        aligned_points,
        target,
        thresholds=thresholds,
        coverage_radius=coverage_radius,
        max_points=max_points,
        voxel_size=voxel_size,
        seed=seed,
    )
    return AlignmentMetricResult(seed=seed, metrics=metrics, alignment=diagnostics)


def identical_cloud_gain_check(
    points: Tensor,
    target: Tensor,
    alignment: str,
    seed: int,
    thresholds: tuple[float, ...],
    coverage_radius: float,
    max_points: int | None,
    voxel_size: float | None,
    trim_fraction: float = 1.0,
    inlier_threshold: float | None = None,
) -> dict[str, Any]:
    baseline = evaluate_alignment_and_metrics(
        points=points,
        target=target,
        alignment=alignment,
        seed=seed,
        thresholds=thresholds,
        coverage_radius=coverage_radius,
        max_points=max_points,
        voxel_size=voxel_size,
        trim_fraction=trim_fraction,
        inlier_threshold=inlier_threshold,
    )
    candidate = evaluate_alignment_and_metrics(
        points=points,
        target=target,
        alignment=alignment,
        seed=seed,
        thresholds=thresholds,
        coverage_radius=coverage_radius,
        max_points=max_points,
        voxel_size=voxel_size,
        trim_fraction=trim_fraction,
        inlier_threshold=inlier_threshold,
    )
    gains = flatten_gains(compute_metric_gains(baseline.metrics, candidate.metrics))
    max_abs_gain = max((abs(value) for value in gains.values()), default=0.0)
    return {
        "seed": seed,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "gains": gains,
        "max_abs_gain": max_abs_gain,
    }


def _summarize_values(values: list[float]) -> dict[str, float | None]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(finite_values),
        "std": pstdev(finite_values),
        "min": min(finite_values),
        "max": max(finite_values),
    }


def summarize_metric_stability(results: list[AlignmentMetricResult]) -> dict[str, Any]:
    metric_names = sorted({
        name for result in results for name in flatten_metrics(result.metrics)
    })
    metric_summary = {
        name: _summarize_values([
            flatten_metrics(result.metrics)[name]
            for result in results
            if name in flatten_metrics(result.metrics)
        ])
        for name in metric_names
    }

    alignment_values: dict[str, list[float]] = {
        "scale": [],
        "rotation_angle_degrees": [],
        "translation_norm": [],
        "residual_mean": [],
        "residual_median": [],
        "residual_rmse": [],
        "residual_max": [],
        "inlier_ratio": [],
    }
    for result in results:
        if result.alignment is None:
            continue
        transform = result.alignment.transform
        alignment_values["scale"].append(transform.scale)
        alignment_values["rotation_angle_degrees"].append(transform.rotation_angle_degrees)
        alignment_values["translation_norm"].append(transform.translation_norm)
        alignment_values["residual_mean"].append(result.alignment.residual_mean)
        alignment_values["residual_median"].append(result.alignment.residual_median)
        alignment_values["residual_rmse"].append(result.alignment.residual_rmse)
        alignment_values["residual_max"].append(result.alignment.residual_max)
        if result.alignment.inlier_ratio is not None:
            alignment_values["inlier_ratio"].append(result.alignment.inlier_ratio)

    return {
        "seed_count": len(results),
        "metrics": metric_summary,
        "alignment": {
            name: _summarize_values(values)
            for name, values in alignment_values.items()
            if values
        },
    }
