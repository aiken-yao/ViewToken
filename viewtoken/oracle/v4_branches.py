"""v4 reconstruction cache loading and branch recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .camera_alignment import (
    CameraAnchorAlignment,
    camera_centers_from_camera_to_world,
    camera_centers_from_world_to_camera,
    decode_vggt_pose_enc,
    estimate_camera_anchor_alignment,
    infer_image_size_hw,
    load_pose_enc,
)
from .cache import validate_reconstruction_cache_v4
from .depth_branch import depth_views_to_known_world_points, normalize_depth_stack, normalize_intrinsics_stack
from .io import load_point_cloud, load_pose_matrix, view_id_from_path
from .known_pose import (
    local_camera_points_to_known_world,
    predicted_world_to_local_camera_points,
    reshape_flattened_points_by_view,
)
from .metrics import apply_similarity_transform, nearest_neighbor_squared_distances, sample_points


@dataclass(frozen=True)
class V4CacheData:
    reconstruction_dir: Path
    metadata: dict[str, Any]
    image_view_ids: list[str]
    points: Tensor
    confidence: Tensor
    per_view_world_points: Tensor
    per_view_confidence: Tensor
    world_to_camera_extrinsics: Tensor
    predicted_intrinsics: Tensor
    depth: Tensor
    depth_conf: Tensor
    transformed_gt_intrinsics: Tensor

    @property
    def view_count(self) -> int:
        return len(self.image_view_ids)

    @property
    def height(self) -> int:
        return int(self.depth.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth.shape[2])


def image_view_ids_from_metadata(metadata: dict[str, Any]) -> list[str]:
    image_paths = metadata.get("image_paths")
    if not isinstance(image_paths, list):
        raise ValueError("metadata image_paths must be a list")
    return [view_id_from_path(Path(path)) for path in image_paths]


def load_gt_poses_for_view_ids(view_ids: list[str], pose_dir: Path) -> Tensor:
    poses = [torch.tensor(load_pose_matrix(pose_dir / f"{view_id}.txt"), dtype=torch.float32) for view_id in view_ids]
    return torch.stack(poses, dim=0)


def load_v4_cache_data(
    reconstruction_dir: Path,
    expected_fingerprint: str | None = None,
    expected_view_ids: list[str] | None = None,
) -> V4CacheData:
    reconstruction_dir = reconstruction_dir.expanduser().resolve()
    metadata = validate_reconstruction_cache_v4(
        reconstruction_dir,
        expected_fingerprint=expected_fingerprint,
    )
    image_view_ids = image_view_ids_from_metadata(metadata)
    if expected_view_ids is not None and image_view_ids != [str(view_id) for view_id in expected_view_ids]:
        raise ValueError(f"cache image order mismatch: expected {expected_view_ids}, got {image_view_ids}")

    points = load_point_cloud(reconstruction_dir / "points.pt")
    confidence = torch.load(reconstruction_dir / "confidence.pt", map_location="cpu", weights_only=True).float()
    per_view_world_points, flatten = reshape_flattened_points_by_view(points, metadata)
    if confidence.numel() != flatten.expected_point_count:
        raise ValueError(
            f"confidence.pt shape does not match metadata: expected {flatten.expected_point_count}, got {confidence.numel()}"
        )
    per_view_confidence = confidence.reshape(flatten.view_count, flatten.height, flatten.width)
    pose_enc = load_pose_enc(reconstruction_dir / "pose_enc.pt")
    extrinsics, decoded_intrinsics = decode_vggt_pose_enc(
        pose_enc,
        image_size_hw=infer_image_size_hw(metadata),
        build_intrinsics=True,
    )
    extrinsics = extrinsics.squeeze(0).float().cpu()
    decoded_intrinsics = decoded_intrinsics.squeeze(0).float().cpu()
    predicted_intrinsics = normalize_intrinsics_stack(
        torch.load(reconstruction_dir / "predicted_intrinsics.pt", map_location="cpu", weights_only=True),
        view_count=flatten.view_count,
    )
    if not torch.allclose(predicted_intrinsics, decoded_intrinsics, rtol=1e-4, atol=1e-4):
        raise ValueError("predicted_intrinsics.pt does not match pose_enc-decoded intrinsics")
    depth = normalize_depth_stack(
        torch.load(reconstruction_dir / "depth.pt", map_location="cpu", weights_only=True),
        label="depth.pt",
    )
    depth_conf = normalize_depth_stack(
        torch.load(reconstruction_dir / "depth_conf.pt", map_location="cpu", weights_only=True),
        label="depth_conf.pt",
    )
    transformed_gt_intrinsics = normalize_intrinsics_stack(
        torch.load(reconstruction_dir / "transformed_gt_intrinsics.pt", map_location="cpu", weights_only=True),
        view_count=flatten.view_count,
    )
    if depth.shape != (flatten.view_count, flatten.height, flatten.width):
        raise ValueError(
            f"depth shape does not match per-view point layout: expected {(flatten.view_count, flatten.height, flatten.width)}, got {tuple(depth.shape)}"
        )
    if depth_conf.shape != depth.shape:
        raise ValueError(f"depth_conf shape mismatch: expected {tuple(depth.shape)}, got {tuple(depth_conf.shape)}")
    return V4CacheData(
        reconstruction_dir=reconstruction_dir,
        metadata=metadata,
        image_view_ids=image_view_ids,
        points=points.float().cpu(),
        confidence=confidence.float().cpu(),
        per_view_world_points=per_view_world_points.float().cpu(),
        per_view_confidence=per_view_confidence.float().cpu(),
        world_to_camera_extrinsics=extrinsics,
        predicted_intrinsics=predicted_intrinsics,
        depth=depth,
        depth_conf=depth_conf,
        transformed_gt_intrinsics=transformed_gt_intrinsics,
    )


def estimate_observed_anchor_alignment_for_cache(
    image_view_ids: list[str],
    world_to_camera_extrinsics: Tensor,
    gt_camera_to_world_poses: Tensor,
    observed_ids: list[str],
) -> CameraAnchorAlignment:
    predicted_centers = camera_centers_from_world_to_camera(world_to_camera_extrinsics).squeeze(0).float().cpu()
    gt_centers = camera_centers_from_camera_to_world(gt_camera_to_world_poses).float().cpu()
    predicted_by_view = {
        view_id: predicted_centers[index]
        for index, view_id in enumerate(image_view_ids)
    }
    gt_by_view = {view_id: gt_centers[index] for index, view_id in enumerate(image_view_ids)}
    return estimate_camera_anchor_alignment(
        predicted_centers_by_view=predicted_by_view,
        gt_centers_by_view=gt_by_view,
        shared_anchor_ids=[str(view_id) for view_id in observed_ids],
    )


def recover_v4_branch_points(
    cache: V4CacheData,
    gt_camera_to_world_poses: Tensor,
    observed_ids: list[str],
    branch: str,
) -> tuple[Tensor, dict[str, Any]]:
    branch = branch.upper()
    gt_camera_to_world_poses = torch.as_tensor(gt_camera_to_world_poses, dtype=torch.float32).cpu()
    if gt_camera_to_world_poses.shape != (cache.view_count, 4, 4):
        raise ValueError(
            f"gt_camera_to_world_poses must have shape {(cache.view_count, 4, 4)}, got {tuple(gt_camera_to_world_poses.shape)}"
        )
    alignment = estimate_observed_anchor_alignment_for_cache(
        image_view_ids=cache.image_view_ids,
        world_to_camera_extrinsics=cache.world_to_camera_extrinsics,
        gt_camera_to_world_poses=gt_camera_to_world_poses,
        observed_ids=observed_ids,
    )
    if branch == "A":
        points = apply_similarity_transform(
            cache.points,
            scale=alignment.transform.scale,
            rotation=alignment.transform.rotation,
            translation=alignment.transform.translation,
        )
        valid = torch.isfinite(points).all(dim=-1)
        return points[valid].contiguous(), {
            "branch": "A",
            "method": "world_points_plus_predicted_pose_observed_camera_anchor_sim3",
            "alignment": alignment.to_dict(),
            "depth_scale": alignment.transform.scale,
            "valid_point_count": int(valid.sum().item()),
        }
    if branch == "B":
        local = predicted_world_to_local_camera_points(
            cache.per_view_world_points,
            cache.world_to_camera_extrinsics,
        )
        world = local_camera_points_to_known_world(
            local,
            gt_camera_to_world_poses,
            depth_scale=alignment.transform.scale,
        )
        flat = world.reshape(-1, 3)
        valid = torch.isfinite(flat).all(dim=-1)
        return flat[valid].contiguous(), {
            "branch": "B",
            "method": "point_head_local_geometry_known_pose_observed_anchor_scale",
            "alignment": alignment.to_dict(),
            "depth_scale": alignment.transform.scale,
            "valid_point_count": int(valid.sum().item()),
        }
    if branch == "C":
        world, valid_grid = depth_views_to_known_world_points(
            cache.depth,
            cache.predicted_intrinsics,
            gt_camera_to_world_poses,
            depth_scale=alignment.transform.scale,
        )
        flat = world.reshape(-1, 3)
        valid = valid_grid.reshape(-1) & torch.isfinite(flat).all(dim=-1)
        return flat[valid].contiguous(), {
            "branch": "C",
            "method": "depth_head_predicted_intrinsics_known_pose_observed_anchor_scale",
            "alignment": alignment.to_dict(),
            "depth_scale": alignment.transform.scale,
            "valid_point_count": int(valid.sum().item()),
        }
    if branch == "D":
        world, valid_grid = depth_views_to_known_world_points(
            cache.depth,
            cache.transformed_gt_intrinsics,
            gt_camera_to_world_poses,
            depth_scale=alignment.transform.scale,
        )
        flat = world.reshape(-1, 3)
        valid = valid_grid.reshape(-1) & torch.isfinite(flat).all(dim=-1)
        return flat[valid].contiguous(), {
            "branch": "D",
            "method": "depth_head_transformed_calibrated_intrinsics_known_pose_observed_anchor_scale",
            "alignment": alignment.to_dict(),
            "depth_scale": alignment.transform.scale,
            "valid_point_count": int(valid.sum().item()),
        }
    raise ValueError(f"Unsupported branch: {branch}")


def heldout_candidate_pose_diagnostics(
    cache: V4CacheData,
    gt_camera_to_world_poses: Tensor,
    observed_ids: list[str],
    candidate_view_id: str,
) -> dict[str, Any]:
    """Evaluate a candidate pose without using it as an alignment anchor."""

    if candidate_view_id not in cache.image_view_ids:
        raise ValueError(f"candidate {candidate_view_id} missing from cache")
    poses = torch.as_tensor(gt_camera_to_world_poses, dtype=torch.float32).cpu()
    alignment = estimate_observed_anchor_alignment_for_cache(
        cache.image_view_ids,
        cache.world_to_camera_extrinsics,
        poses,
        observed_ids,
    )
    index = cache.image_view_ids.index(candidate_view_id)
    predicted_centers = camera_centers_from_world_to_camera(cache.world_to_camera_extrinsics).squeeze(0)
    aligned_center = alignment.transform.apply(predicted_centers[index : index + 1])[0]
    gt_center = poses[index, :3, 3]

    predicted_camera_to_world_rotation = cache.world_to_camera_extrinsics[index, :3, :3].T
    predicted_forward = predicted_camera_to_world_rotation[:, 2]
    aligned_forward = alignment.transform.rotation @ predicted_forward
    gt_forward = poses[index, :3, 2]
    cosine = torch.dot(aligned_forward, gt_forward) / (
        torch.linalg.norm(aligned_forward) * torch.linalg.norm(gt_forward)
    ).clamp_min(1e-12)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    return {
        "candidate_view_id": candidate_view_id,
        "alignment_uses_observed_ids_only": list(alignment.shared_anchor_ids),
        "predicted_center_before_alignment": predicted_centers[index].tolist(),
        "predicted_center_after_alignment": aligned_center.tolist(),
        "gt_center": gt_center.tolist(),
        "center_error_meters": float(torch.linalg.norm(aligned_center - gt_center).item()),
        "orientation_error_degrees": float(angle.item()),
        "alignment": alignment.to_dict(),
    }


def _distance_summary(values: Tensor) -> dict[str, Any]:
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "max": float(values.max().item()),
    }


def compare_branch_reconstructions(
    target_points: Tensor,
    baseline_points: Tensor,
    candidate_points: Tensor,
    observed_mask: Tensor,
    novel_mask: Tensor,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
    outlier_threshold: float = 0.10,
    max_prediction_points: int | None = 12000,
    seed: int = 0,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Compare baseline/candidate geometry on one shared target and masks."""

    target = torch.as_tensor(target_points, dtype=torch.float32).cpu()
    observed = torch.as_tensor(observed_mask, dtype=torch.bool).flatten().cpu()
    novel = torch.as_tensor(novel_mask, dtype=torch.bool).flatten().cpu()
    if target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError(f"target_points must have shape [N, 3], got {tuple(target.shape)}")
    if observed.numel() != target.shape[0] or novel.numel() != target.shape[0]:
        raise ValueError("visibility masks must match target point count")
    baseline = sample_points(baseline_points, max_prediction_points, seed=seed + 101, method="hash")
    candidate = sample_points(candidate_points, max_prediction_points, seed=seed + 101, method="hash")

    target_to_baseline = nearest_neighbor_squared_distances(target, baseline, chunk_size=chunk_size).sqrt()
    target_to_candidate = nearest_neighbor_squared_distances(target, candidate, chunk_size=chunk_size).sqrt()
    baseline_to_target = nearest_neighbor_squared_distances(baseline, target, chunk_size=chunk_size).sqrt()
    candidate_to_target = nearest_neighbor_squared_distances(candidate, target, chunk_size=chunk_size).sqrt()

    def region_payload(mask: Tensor) -> dict[str, Any]:
        before = target_to_baseline[mask]
        after = target_to_candidate[mask]
        covered = {}
        for threshold in thresholds:
            key = f"{threshold:g}"
            before_count = int((before <= threshold).sum().item())
            after_count = int((after <= threshold).sum().item())
            total = int(mask.sum().item())
            covered[key] = {
                "baseline_count": before_count,
                "candidate_count": after_count,
                "covered_count_gain": after_count - before_count,
                "baseline_ratio": None if total == 0 else before_count / total,
                "candidate_ratio": None if total == 0 else after_count / total,
                "ratio_gain": None if total == 0 else (after_count - before_count) / total,
            }
        return {
            "target_count": int(mask.sum().item()),
            "baseline_distance": _distance_summary(before),
            "candidate_distance": _distance_summary(after),
            "covered": covered,
        }

    return {
        "baseline_point_count": int(baseline.shape[0]),
        "candidate_point_count": int(candidate.shape[0]),
        "novel": region_payload(novel),
        "observed": region_payload(observed),
        "accuracy": {
            "baseline": _distance_summary(baseline_to_target),
            "candidate": _distance_summary(candidate_to_target),
            "baseline_outlier_ratio": float((baseline_to_target > outlier_threshold).float().mean().item()),
            "candidate_outlier_ratio": float((candidate_to_target > outlier_threshold).float().mean().item()),
            "outlier_ratio_gain": float(
                (baseline_to_target > outlier_threshold).float().mean().item()
                - (candidate_to_target > outlier_threshold).float().mean().item()
            ),
        },
        "identical_input": bool(
            baseline.shape == candidate.shape and torch.equal(baseline, candidate)
        ),
    }


def tensor_stats(tensor: Tensor) -> dict[str, Any]:
    values = torch.as_tensor(tensor, dtype=torch.float32).flatten().cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"count": int(values.numel()), "finite_ratio": 0.0, "min": None, "median": None, "max": None}
    return {
        "count": int(values.numel()),
        "finite_ratio": float(finite.numel()) / float(values.numel()),
        "min": float(finite.min().item()),
        "median": float(finite.median().item()),
        "max": float(finite.max().item()),
    }


def cache_artifact_shape_summary(cache: V4CacheData) -> dict[str, Any]:
    return {
        "cache_dir": str(cache.reconstruction_dir),
        "image_view_ids": cache.image_view_ids,
        "input_shape": cache.metadata.get("input_shape"),
        "points": list(cache.points.shape),
        "confidence": list(cache.confidence.shape),
        "pose_enc": [1, cache.view_count, 9],
        "depth": list(cache.depth.shape),
        "depth_conf": list(cache.depth_conf.shape),
        "predicted_intrinsics": list(cache.predicted_intrinsics.shape),
        "transformed_gt_intrinsics": list(cache.transformed_gt_intrinsics.shape),
        "per_view_shape_offsets": cache.metadata.get("per_view_shape_offsets"),
        "depth_stats": tensor_stats(cache.depth),
        "depth_conf_stats": tensor_stats(cache.depth_conf),
    }


def candidate_view_depth_diagnostics(cache: V4CacheData, candidate_view_id: str) -> dict[str, Any]:
    if candidate_view_id not in cache.image_view_ids:
        raise ValueError(f"candidate {candidate_view_id} missing from cache image order {cache.image_view_ids}")
    index = cache.image_view_ids.index(candidate_view_id)
    depth = cache.depth[index]
    depth_conf = cache.depth_conf[index]
    predicted = cache.predicted_intrinsics[index]
    calibrated = cache.transformed_gt_intrinsics[index]
    intrinsics_delta = predicted - calibrated
    positive = torch.isfinite(depth) & (depth > 0.0)
    return {
        "candidate_view_id": candidate_view_id,
        "candidate_index": index,
        "depth": tensor_stats(depth),
        "depth_conf": tensor_stats(depth_conf),
        "positive_depth_ratio": float(positive.float().mean().item()),
        "predicted_intrinsics": predicted.tolist(),
        "transformed_calibrated_intrinsics": calibrated.tolist(),
        "predicted_minus_calibrated_intrinsics": intrinsics_delta.tolist(),
        "intrinsics_abs_delta_max": float(intrinsics_delta.abs().max().item()),
        "intrinsics_focal_ratio_fx": None if abs(float(calibrated[0, 0])) < 1e-12 else float(predicted[0, 0] / calibrated[0, 0]),
        "intrinsics_focal_ratio_fy": None if abs(float(calibrated[1, 1])) < 1e-12 else float(predicted[1, 1] / calibrated[1, 1]),
    }
