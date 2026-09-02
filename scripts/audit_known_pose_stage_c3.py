"""Stage-C3 known-pose diagnostic using existing complete v3 caches only."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_gt_visibility_stage_c2 import (  # noqa: E402
    alignment_transform,
    build_visibility_masks,
    collect_invalid_reconstruction_caches,
    compute_visibility_metrics,
    load_aligned_points,
    load_intrinsics,
    load_jsonl,
    load_pose,
    nearest_distances,
    prepare_metric_points,
    read_image_size,
    resolve_path,
    summarize_metric_values,
    summarize_visibility_masks,
    write_json,
    write_jsonl,
)
from viewtoken.oracle import (  # noqa: E402
    KnownPoseCacheEligibilityError,
    camera_centers_from_world_to_camera,
    decode_vggt_pose_enc,
    fuse_cached_points_with_known_poses,
    infer_image_size_hw,
    load_point_cloud,
    load_pose_enc,
    validate_reconstruction_cache,
)
from viewtoken.oracle.io import view_id_from_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audit-records", type=Path, default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--posed-image-dir", type=Path, default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--c2-variant-records", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return payload


def value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    cli_value = getattr(args, name.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def optional_path(args: argparse.Namespace, config: dict[str, Any], name: str) -> Path | None:
    raw = value(args, config, name, None)
    if raw is None:
        return None
    return resolve_path(raw, name)


def percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.float().cpu(), q).item())


def distance_distribution(distances: torch.Tensor) -> dict[str, Any]:
    distances = distances.float().cpu()
    if distances.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": int(distances.numel()),
        "min": float(distances.min().item()),
        "p10": percentile(distances, 0.10),
        "p25": percentile(distances, 0.25),
        "median": percentile(distances, 0.50),
        "p75": percentile(distances, 0.75),
        "p90": percentile(distances, 0.90),
        "p95": percentile(distances, 0.95),
        "mean": float(distances.mean().item()),
        "max": float(distances.max().item()),
    }


def covered_payload(distances: torch.Tensor, thresholds: tuple[float, ...]) -> dict[str, Any]:
    distances = distances.float().cpu()
    payload = {"total_count": int(distances.numel()), "thresholds": {}}
    for threshold in thresholds:
        key = f"{threshold:g}"
        if distances.numel() == 0:
            payload["thresholds"][key] = {"covered_count": 0, "covered_ratio": None}
        else:
            covered = int((distances <= float(threshold)).sum().item())
            payload["thresholds"][key] = {
                "covered_count": covered,
                "covered_ratio": float(covered) / float(distances.numel()),
            }
    return payload


def branch_geometry_metrics(
    target_points: torch.Tensor,
    baseline_points: torch.Tensor,
    candidate_points: torch.Tensor,
    masks: Any,
    coverage_thresholds: tuple[float, ...],
    outlier_threshold: float,
    chunk_size: int,
) -> dict[str, Any]:
    target_to_baseline = nearest_distances(target_points, baseline_points, chunk_size=chunk_size)
    target_to_candidate = nearest_distances(target_points, candidate_points, chunk_size=chunk_size)
    baseline_to_target = nearest_distances(baseline_points, target_points, chunk_size=chunk_size)
    candidate_to_target = nearest_distances(candidate_points, target_points, chunk_size=chunk_size)

    metrics = compute_visibility_metrics(
        target_distances_to_baseline=target_to_baseline,
        target_distances_to_candidate=target_to_candidate,
        baseline_distances_to_target=baseline_to_target,
        candidate_distances_to_target=candidate_to_target,
        masks=masks,
        coverage_thresholds=coverage_thresholds,
        outlier_threshold=outlier_threshold,
    )
    novel_mask = masks.novel
    baseline_novel_distances = target_to_baseline[novel_mask]
    candidate_novel_distances = target_to_candidate[novel_mask]
    raw_coverage = {
        "baseline": covered_payload(baseline_novel_distances, coverage_thresholds),
        "candidate": covered_payload(candidate_novel_distances, coverage_thresholds),
        "gain": {},
    }
    scene_count = int(target_points.shape[0])
    novel_count = int(novel_mask.sum().item())
    for threshold in coverage_thresholds:
        key = f"{threshold:g}"
        baseline_count = raw_coverage["baseline"]["thresholds"][key]["covered_count"]
        candidate_count = raw_coverage["candidate"]["thresholds"][key]["covered_count"]
        raw_coverage["gain"][key] = {
            "covered_count_gain": int(candidate_count - baseline_count),
            "novel_ratio_gain": None
            if novel_count == 0
            else float(candidate_count - baseline_count) / float(novel_count),
            "scene_normalized_gain": float(candidate_count - baseline_count) / float(scene_count),
        }

    return {
        "visibility_metrics": metrics,
        "novel_distance_distribution": {
            "target_to_baseline": distance_distribution(baseline_novel_distances),
            "target_to_candidate": distance_distribution(candidate_novel_distances),
        },
        "novel_raw_coverage": raw_coverage,
    }


def cache_metadata(record: dict[str, Any], role: str) -> dict[str, Any]:
    path = Path(record["reconstruction_paths"][role]).expanduser().resolve()
    expected = record.get("metadata", {}).get(f"{role}_reconstruction", {}).get("cache_fingerprint")
    return validate_reconstruction_cache(path.parent, expected_fingerprint=expected)


def reconstruction_dir(record: dict[str, Any], role: str) -> Path:
    return Path(record["reconstruction_paths"][role]).expanduser().resolve().parent


def decode_cache_extrinsics(record: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata = cache_metadata(record, role)
    pose_enc = load_pose_enc(reconstruction_dir(record, role) / "pose_enc.pt")
    extrinsics, _intrinsics = decode_vggt_pose_enc(
        pose_enc,
        image_size_hw=infer_image_size_hw(metadata),
        build_intrinsics=False,
    )
    return extrinsics.squeeze(0).float().cpu(), metadata


def load_cache_image_view_ids(metadata: dict[str, Any]) -> list[str]:
    return [view_id_from_path(Path(path)) for path in metadata["image_paths"]]


def load_gt_poses_for_cache(metadata: dict[str, Any], posed_image_dir: Path) -> torch.Tensor:
    poses = []
    for image_path in metadata["image_paths"]:
        view_id = view_id_from_path(Path(image_path))
        poses.append(load_pose(posed_image_dir / f"{view_id}.txt"))
    return torch.stack(poses, dim=0)


def known_pose_branch_points(
    record: dict[str, Any],
    role: str,
    posed_image_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata = cache_metadata(record, role)
    confidence = torch.load(
        reconstruction_dir(record, role) / "confidence.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    points = load_point_cloud(Path(record["reconstruction_paths"][role]).expanduser().resolve())
    extrinsics, _metadata = decode_cache_extrinsics(record, role)
    poses = load_gt_poses_for_cache(metadata, posed_image_dir)
    transform = alignment_transform(record, role)
    fused, flatten = fuse_cached_points_with_known_poses(
        points,
        metadata,
        world_to_camera_extrinsics=extrinsics,
        camera_to_world_poses=poses,
        depth_scale=transform.scale,
    )
    expected = int(flatten["expected_point_count"])
    if int(confidence.numel()) != expected:
        raise KnownPoseCacheEligibilityError(
            f"confidence.pt shape does not match expected point count: {confidence.numel()} vs {expected}"
        )
    return fused, {
        "cache_dir": str(reconstruction_dir(record, role)),
        "image_view_ids": load_cache_image_view_ids(metadata),
        "flatten": flatten,
        "depth_scale_from_observed_anchor_sim3": transform.scale,
        "method": "cached_v3_world_points_to_local_camera_then_gt_camera_to_world",
    }


def rotation_angle_degrees(rotation: torch.Tensor) -> float:
    trace = torch.trace(rotation.float()).item()
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def angle_between(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().cpu()
    right = right.float().cpu()
    left = left / torch.linalg.norm(left).clamp_min(1e-12)
    right = right / torch.linalg.norm(right).clamp_min(1e-12)
    cosine = torch.dot(left, right).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def pairwise_distance_payload(
    predicted_centers: torch.Tensor,
    aligned_centers: torch.Tensor,
    gt_centers: torch.Tensor,
    view_ids: list[str],
) -> dict[str, Any]:
    rows = []
    abs_errors = []
    rel_errors = []
    raw_ratios = []
    for left in range(len(view_ids)):
        for right in range(left + 1, len(view_ids)):
            gt_distance = torch.linalg.norm(gt_centers[left] - gt_centers[right]).item()
            aligned_distance = torch.linalg.norm(aligned_centers[left] - aligned_centers[right]).item()
            raw_distance = torch.linalg.norm(predicted_centers[left] - predicted_centers[right]).item()
            abs_error = aligned_distance - gt_distance
            rel_error = None if gt_distance <= 1e-12 else abs_error / gt_distance
            raw_ratio = None if gt_distance <= 1e-12 else raw_distance / gt_distance
            rows.append(
                {
                    "pair": [view_ids[left], view_ids[right]],
                    "gt_distance": gt_distance,
                    "predicted_raw_distance": raw_distance,
                    "aligned_predicted_distance": aligned_distance,
                    "aligned_minus_gt": abs_error,
                    "relative_error": rel_error,
                    "raw_predicted_to_gt_ratio": raw_ratio,
                }
            )
            abs_errors.append(abs(abs_error))
            if rel_error is not None:
                rel_errors.append(abs(rel_error))
            if raw_ratio is not None:
                raw_ratios.append(raw_ratio)
    return {
        "pairs": rows,
        "mean_abs_error_meters": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "max_abs_error_meters": max(abs_errors) if abs_errors else None,
        "mean_abs_relative_error": sum(rel_errors) / len(rel_errors) if rel_errors else None,
        "max_abs_relative_error": max(rel_errors) if rel_errors else None,
        "mean_raw_predicted_to_gt_ratio": sum(raw_ratios) / len(raw_ratios) if raw_ratios else None,
    }


def heldout_pose_diagnostics(
    record: dict[str, Any],
    posed_image_dir: Path,
) -> dict[str, Any]:
    candidate_id = str(record["candidate_view_id"])
    observed_ids = list(record["observed_view_ids"])
    extrinsics, metadata = decode_cache_extrinsics(record, "candidate")
    image_view_ids = load_cache_image_view_ids(metadata)
    candidate_index = len(image_view_ids) - 1
    shared_indices = list(range(len(observed_ids))) + [candidate_index]
    compared_view_ids = observed_ids + [candidate_id]

    centers = camera_centers_from_world_to_camera(extrinsics).squeeze(0).float().cpu()
    transform = alignment_transform(record, "candidate")
    aligned_centers = transform.apply(centers)
    gt_poses = torch.stack(
        [load_pose(posed_image_dir / f"{view_id}.txt") for view_id in compared_view_ids],
        dim=0,
    )
    gt_centers = gt_poses[:, :3, 3].float().cpu()
    candidate_center_error = torch.linalg.norm(
        aligned_centers[candidate_index] - gt_centers[-1]
    ).item()

    predicted_rotation_c2w = extrinsics[candidate_index, :3, :3].T
    aligned_rotation_c2w = transform.rotation.float().cpu() @ predicted_rotation_c2w
    gt_rotation_c2w = gt_poses[-1, :3, :3].float().cpu()
    rotation_error = rotation_angle_degrees(aligned_rotation_c2w.T @ gt_rotation_c2w)
    forward_error = angle_between(aligned_rotation_c2w[:, 2], gt_rotation_c2w[:, 2])

    return {
        "candidate_view_id": candidate_id,
        "candidate_cache_image_view_ids": image_view_ids,
        "candidate_index_in_cache": candidate_index,
        "candidate_used_as_alignment_anchor": False,
        "shared_observed_anchor_ids": observed_ids,
        "heldout_candidate_center_error_meters": candidate_center_error,
        "heldout_candidate_rotation_error_degrees": rotation_error,
        "heldout_candidate_forward_error_degrees": forward_error,
        "pairwise_distance_distortion": pairwise_distance_payload(
            predicted_centers=centers[shared_indices],
            aligned_centers=aligned_centers[shared_indices],
            gt_centers=gt_centers,
            view_ids=compared_view_ids,
        ),
    }


def semantic_label(row: dict[str, Any], min_largest_component: int) -> str:
    stats = row["visibility_stats"]
    novel_count = int(stats["novel_count"])
    overlap_count = int(stats["overlap_count"])
    largest_component = int(row.get("novel_connectivity", {}).get("largest_component_count", 0))
    candidate_overlap = stats["candidate_overlap_fraction"] or 0.0
    if novel_count == 0:
        return "duplicate_or_no_novel"
    if overlap_count == 0:
        return "disconnected_novel_view"
    if largest_component >= min_largest_component:
        return "connected_novel_view"
    if candidate_overlap >= 0.95:
        return "high_overlap_low_novel"
    return "mixed_overlap_novel_view"


def overlap_stability(c2_variant_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in c2_variant_records:
        by_candidate.setdefault(str(row["candidate_view_id"]), []).append(row)
    return {
        candidate_id: {
            "variant_count": len(rows),
            "candidate_overlap_fraction": summarize_metric_values(
                [row["visibility_stats"]["candidate_overlap_fraction"] for row in rows]
            ),
            "scene_overlap_fraction": summarize_metric_values(
                [row["visibility_stats"]["overlap_fraction"] for row in rows]
            ),
        }
        for candidate_id, rows in sorted(by_candidate.items())
    }


def branch_summary(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    return {
        "novel_scene_normalized_gain_0.05": summarize_metric_values(
            [
                row["branches"][branch]["novel_raw_coverage"]["gain"]["0.05"]["scene_normalized_gain"]
                for row in rows
            ]
        ),
        "novel_scene_normalized_gain_0.10": summarize_metric_values(
            [
                row["branches"][branch]["novel_raw_coverage"]["gain"]["0.1"]["scene_normalized_gain"]
                for row in rows
            ]
        ),
        "observed_retention_gain_0.05": summarize_metric_values(
            [
                row["branches"][branch]["visibility_metrics"]["observed_retention_gain"]["0.05"]["gain"]
                for row in rows
            ]
        ),
        "global_accuracy_gain": summarize_metric_values(
            [row["branches"][branch]["visibility_metrics"]["global_accuracy"]["gain"] for row in rows]
        ),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for row in summary["candidate_rows"]:
        stats = row["visibility_stats"]
        pose = row["heldout_pose_diagnostics"]
        pred = row["branches"]["predicted_world"]["novel_raw_coverage"]
        known = row["branches"]["known_pose"]["novel_raw_coverage"]
        pred_metrics = row["branches"]["predicted_world"]["visibility_metrics"]
        known_metrics = row["branches"]["known_pose"]["visibility_metrics"]
        pairwise = pose["pairwise_distance_distortion"]
        rows.append(
            "| `{cand}` | `{sem}` | `{overlap:.6f}` | `{novel:.6f}` | `{pose_err:.4f}` | "
            "`{rot_err:.2f}` | `{pair_mean:.4f}`/`{pair_max:.4f}` | `{pred05}`/`{pred10}` | `{known05}`/`{known10}` | "
            "`{pred_ret:.6f}` | `{known_ret:.6f}` | `{known_acc:.6f}` |".format(
                cand=row["candidate_view_id"],
                sem=row["corrected_visibility_semantic_tag"],
                overlap=stats["candidate_overlap_fraction"] or 0.0,
                novel=stats["novel_scene_fraction"],
                pose_err=pose["heldout_candidate_center_error_meters"],
                rot_err=pose["heldout_candidate_rotation_error_degrees"],
                pair_mean=pairwise["mean_abs_error_meters"] or 0.0,
                pair_max=pairwise["max_abs_error_meters"] or 0.0,
                pred05=pred["candidate"]["thresholds"]["0.05"]["covered_count"],
                pred10=pred["candidate"]["thresholds"]["0.1"]["covered_count"],
                known05=known["candidate"]["thresholds"]["0.05"]["covered_count"],
                known10=known["candidate"]["thresholds"]["0.1"]["covered_count"],
                pred_ret=pred_metrics["observed_retention_gain"]["0.05"]["gain"] or 0.0,
                known_ret=known_metrics["observed_retention_gain"]["0.05"]["gain"] or 0.0,
                known_acc=known_metrics["global_accuracy"]["gain"],
            )
        )

    text = f"""# Stage C3 Known-Pose Diagnostic

Status: `{summary['status']}`

Stage C3 passed: `{summary['assessment']['stage_c3_passed']}`

This diagnostic reused existing complete v3 caches and did not run VGGT.

## Scope

- Predicted-world branch: VGGT `world_points` plus observed-camera-anchor Sim(3).
- Known-pose branch: cached v3 `world_points` converted back to per-view local camera coordinates with VGGT predicted extrinsics, scaled by observed-anchor Sim(3), then fused with ScanNet GT camera-to-world poses.
- True v4 depth-backprojection branch status: `{summary['known_pose_branch']['v4_depth_backprojection_status']}`
- Candidate RGB/depth/visibility remains offline audit data only, not future policy input.

## Candidate Summary

| cand | corrected visibility tag | cand overlap frac | novel scene frac | held-out center err m | rot err deg | pairwise err mean/max m | A cand covered novel @0.05/@0.10 | B cand covered novel @0.05/@0.10 | A obs retention@0.05 | B obs retention@0.05 | B global acc gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## C2 Semantic Correction

- `00325` and `00425` have `M_overlap = 0` in the nominal C2 masks, so they are `disconnected_novel_view`, not connected-new-area views.
- A connected-novel candidate must have both `novel_count > 0` and `overlap_count > 0`.
- Overlap stability is recorded across `{summary['overlap_stability_variant_count']}` C2 variants in the JSON report.

## Assessment

- Predicted-world connected candidates with stable positive novel gain: `{summary['assessment']['predicted_world_connected_positive_candidates']}`
- Known-pose connected candidates with stable positive novel gain: `{summary['assessment']['known_pose_connected_positive_candidates']}`
- Disconnected novel candidates: `{summary['assessment']['disconnected_novel_candidates']}`
- Known-pose branch supported by existing v3 cache: `{summary['known_pose_branch']['cached_known_pose_branch_supported']}`

Full JSON summary: `{summary['outputs']['docs_summary_json']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)

    audit_records_path = resolve_path(value(args, config, "audit-records", None), "audit-records")
    target_points_path = resolve_path(value(args, config, "target-points", None), "target-points")
    posed_image_dir = resolve_path(value(args, config, "posed-image-dir", None), "posed-image-dir")
    intrinsics_path = resolve_path(value(args, config, "intrinsics", None), "intrinsics")
    c2_variant_records_path = optional_path(args, config, "c2-variant-records")
    output_dir = resolve_path(
        value(args, config, "output-dir", "outputs/oracle_calibration/scannet_scene0000_00_stage_c3"),
        "output-dir",
        must_exist=False,
    )
    docs_output_dir = resolve_path(
        value(args, config, "docs-output-dir", "docs/audits/scannet_scene0000_00_stage_c3"),
        "docs-output-dir",
        must_exist=False,
    )

    records = load_jsonl(audit_records_path)
    candidate_filter = value(args, config, "candidate-view-ids", None)
    if candidate_filter is not None:
        keep = {str(item) for item in candidate_filter}
        records = [record for record in records if str(record["candidate_view_id"]) in keep]
    invalid_caches = collect_invalid_reconstruction_caches(records)
    if invalid_caches:
        raise RuntimeError(f"Invalid v3 reconstruction caches: {invalid_caches}")

    image_extension = str(value(args, config, "image-extension", "jpg"))
    image_size_wh = read_image_size(
        posed_image_dir / f"{records[0]['observed_view_ids'][0]}.{image_extension}"
    )
    intrinsics = load_intrinsics(intrinsics_path, image_size_wh=image_size_wh)
    observed_ids = list(records[0]["observed_view_ids"])
    observed_poses = {
        view_id: load_pose(posed_image_dir / f"{view_id}.txt")
        for view_id in observed_ids
    }

    target_points = load_point_cloud(
        target_points_path,
        point_stride=int(value(args, config, "point-stride", 6)),
    )
    voxel_size = float(value(args, config, "gt-voxel-size", 0.02))
    depth_tolerance = float(value(args, config, "visibility-depth-tolerance", 0.05))
    seed = int(value(args, config, "seed", 0))
    max_surface_points = int(value(args, config, "max-surface-points", 12000))
    max_prediction_points = int(value(args, config, "max-prediction-points", 12000))
    sample_method = str(value(args, config, "sample-method", "hash"))
    prediction_sample_method = str(value(args, config, "prediction-sample-method", sample_method))
    coverage_thresholds = tuple(
        float(item) for item in value(args, config, "coverage-thresholds", [0.05, 0.10, 0.20, 0.50])
    )
    outlier_threshold = float(value(args, config, "global-outlier-threshold", 0.10))
    chunk_size = int(value(args, config, "nn-chunk-size", 512))
    min_largest_component = int(value(args, config, "novel-connectivity-min-largest-component-count", 10))

    target_sample = prepare_metric_points(
        target_points,
        voxel_size=voxel_size,
        max_points=max_surface_points,
        seed=seed,
        method=sample_method,
    )
    predicted_baseline = prepare_metric_points(
        load_aligned_points(records[0], "baseline"),
        voxel_size=voxel_size,
        max_points=max_prediction_points,
        seed=seed + 101,
        method=prediction_sample_method,
    )
    known_baseline_full, known_baseline_metadata = known_pose_branch_points(
        records[0],
        "baseline",
        posed_image_dir=posed_image_dir,
    )
    known_baseline = prepare_metric_points(
        known_baseline_full,
        voxel_size=voxel_size,
        max_points=max_prediction_points,
        seed=seed + 101,
        method=prediction_sample_method,
    )

    candidate_rows = []
    known_pose_failures = []
    for record in records:
        candidate_id = str(record["candidate_view_id"])
        candidate_pose = load_pose(posed_image_dir / f"{candidate_id}.txt")
        masks = build_visibility_masks(
            target_sample,
            observed_poses.values(),
            candidate_pose,
            intrinsics=intrinsics,
            depth_tolerance=depth_tolerance,
        )
        stats = summarize_visibility_masks(masks).to_dict()

        predicted_candidate = prepare_metric_points(
            load_aligned_points(record, "candidate"),
            voxel_size=voxel_size,
            max_points=max_prediction_points,
            seed=seed + 211,
            method=prediction_sample_method,
        )
        predicted_branch = branch_geometry_metrics(
            target_points=target_sample,
            baseline_points=predicted_baseline,
            candidate_points=predicted_candidate,
            masks=masks,
            coverage_thresholds=coverage_thresholds,
            outlier_threshold=outlier_threshold,
            chunk_size=chunk_size,
        )

        try:
            known_candidate_full, known_candidate_metadata = known_pose_branch_points(
                record,
                "candidate",
                posed_image_dir=posed_image_dir,
            )
            known_candidate = prepare_metric_points(
                known_candidate_full,
                voxel_size=voxel_size,
                max_points=max_prediction_points,
                seed=seed + 211,
                method=prediction_sample_method,
            )
            known_branch = branch_geometry_metrics(
                target_points=target_sample,
                baseline_points=known_baseline,
                candidate_points=known_candidate,
                masks=masks,
                coverage_thresholds=coverage_thresholds,
                outlier_threshold=outlier_threshold,
                chunk_size=chunk_size,
            )
            known_metadata = known_candidate_metadata
        except KnownPoseCacheEligibilityError as exc:
            known_pose_failures.append({"candidate_view_id": candidate_id, "error": str(exc)})
            known_branch = {"status": "blocked_known_pose_cache_eligibility", "error": str(exc)}
            known_metadata = {}

        row = {
            "scene_id": record["scene_id"],
            "candidate_view_id": candidate_id,
            "original_candidate_sanity_tags": record.get("metadata", {}).get("candidate_sanity_tags", []),
            "visibility_stats": stats,
            "corrected_visibility_semantic_tag": None,
            "heldout_pose_diagnostics": heldout_pose_diagnostics(record, posed_image_dir),
            "branches": {
                "predicted_world": predicted_branch,
                "known_pose": known_branch,
            },
            "known_pose_cache_metadata": known_metadata,
        }
        row["corrected_visibility_semantic_tag"] = semantic_label(
            row,
            min_largest_component=min_largest_component,
        )
        candidate_rows.append(row)

    if c2_variant_records_path is not None:
        c2_variant_records = load_jsonl(c2_variant_records_path)
        overlap = overlap_stability(c2_variant_records)
        overlap_variant_count = len({json.dumps(row["variant_key"], sort_keys=True) for row in c2_variant_records})
    else:
        overlap = {}
        overlap_variant_count = 0

    semantic_counts = Counter(row["corrected_visibility_semantic_tag"] for row in candidate_rows)
    connected_candidates = [
        row for row in candidate_rows if row["corrected_visibility_semantic_tag"] == "connected_novel_view"
    ]
    predicted_positive = [
        row["candidate_view_id"]
        for row in connected_candidates
        if row["branches"]["predicted_world"]["novel_raw_coverage"]["gain"]["0.05"]["scene_normalized_gain"] > 0.0
        and row["branches"]["predicted_world"]["novel_raw_coverage"]["gain"]["0.1"]["scene_normalized_gain"] > 0.0
    ]
    known_positive = [
        row["candidate_view_id"]
        for row in connected_candidates
        if isinstance(row["branches"]["known_pose"], dict)
        and "novel_raw_coverage" in row["branches"]["known_pose"]
        and row["branches"]["known_pose"]["novel_raw_coverage"]["gain"]["0.05"]["scene_normalized_gain"] > 0.0
        and row["branches"]["known_pose"]["novel_raw_coverage"]["gain"]["0.1"]["scene_normalized_gain"] > 0.0
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "stage_c3_known_pose_records.jsonl"
    output_summary_path = output_dir / "stage_c3_known_pose_summary.json"
    docs_summary_path = docs_output_dir / "stage_c3_known_pose_summary.json"
    markdown_path = docs_output_dir / "run_log.md"

    summary = {
        "status": "complete",
        "did_run_vggt": False,
        "inputs": {
            "audit_records": str(audit_records_path),
            "target_points": str(target_points_path),
            "posed_image_dir": str(posed_image_dir),
            "intrinsics": str(intrinsics_path),
            "c2_variant_records": None if c2_variant_records_path is None else str(c2_variant_records_path),
        },
        "config": {
            "gt_voxel_size": voxel_size,
            "visibility_depth_tolerance": depth_tolerance,
            "max_surface_points": max_surface_points,
            "max_prediction_points": max_prediction_points,
            "sample_method": sample_method,
            "prediction_sample_method": prediction_sample_method,
            "coverage_thresholds": list(coverage_thresholds),
            "global_outlier_threshold": outlier_threshold,
            "novel_connectivity_min_largest_component_count": min_largest_component,
            "intrinsics": intrinsics.to_dict(),
        },
        "candidate_count": len(candidate_rows),
        "candidate_rows": sorted(candidate_rows, key=lambda row: row["candidate_view_id"]),
        "c2_corrected_semantic_counts": dict(sorted(semantic_counts.items())),
        "overlap_stability_variant_count": overlap_variant_count,
        "overlap_stability_by_candidate": overlap,
        "branch_summary": {
            "predicted_world": branch_summary(candidate_rows, "predicted_world"),
            "known_pose": branch_summary(
                [row for row in candidate_rows if "novel_raw_coverage" in row["branches"]["known_pose"]],
                "known_pose",
            ),
        },
        "known_pose_branch": {
            "cached_known_pose_branch_supported": not known_pose_failures,
            "known_pose_failures": known_pose_failures,
            "baseline_cache_metadata": known_baseline_metadata,
            "v4_depth_backprojection_status": (
                "blocked_missing_v4_depth_conf_per_view_offsets_and_preprocessing_transform; "
                "cached v3 diagnostic used local camera coordinates recovered from world_points and pose_enc instead"
            ),
        },
        "assessment": {
            "stage_c3_passed": False,
            "predicted_world_connected_positive_candidates": predicted_positive,
            "known_pose_connected_positive_candidates": known_positive,
            "disconnected_novel_candidates": [
                row["candidate_view_id"]
                for row in candidate_rows
                if row["corrected_visibility_semantic_tag"] == "disconnected_novel_view"
            ],
            "needs_connected_novel_candidate_selection_before_more_vggt": True,
        },
        "outputs": {
            "variant_records_jsonl": str(records_path),
            "output_summary_json": str(output_summary_path),
            "docs_summary_json": str(docs_summary_path),
            "docs_markdown": str(markdown_path),
        },
        "runtime_seconds": time.perf_counter() - started_at,
    }

    write_jsonl(records_path, summary["candidate_rows"])
    write_json(output_summary_path, summary)
    write_json(docs_summary_path, summary)
    write_markdown(markdown_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "stage_c3_passed": summary["assessment"]["stage_c3_passed"],
        "candidate_count": summary["candidate_count"],
        "did_run_vggt": summary["did_run_vggt"],
        "summary": str(docs_summary_path),
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
