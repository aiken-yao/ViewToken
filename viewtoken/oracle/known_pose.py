"""Known-pose fusion diagnostics for cached VGGT geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


class KnownPoseCacheEligibilityError(RuntimeError):
    """Raised when a cache cannot prove per-view geometry ownership."""


@dataclass(frozen=True)
class PerViewFlattenMetadata:
    view_count: int
    height: int
    width: int
    expected_point_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "view_count": self.view_count,
            "height": self.height,
            "width": self.width,
            "expected_point_count": self.expected_point_count,
        }


def inspect_per_view_flatten_metadata(metadata: dict[str, Any]) -> PerViewFlattenMetadata:
    """Validate that points.pt can be reshaped back to [view, H, W, 3]."""

    input_shape = metadata.get("input_shape")
    if not isinstance(input_shape, list | tuple) or len(input_shape) != 4:
        raise KnownPoseCacheEligibilityError(
            f"metadata input_shape must be [S, C, H, W], got {input_shape}"
        )
    view_count, _channels, height, width = [int(value) for value in input_shape]
    expected_count = view_count * height * width

    checks = {
        "raw_world_point_count_before_filter": metadata.get("raw_world_point_count_before_filter"),
        "filtered_world_point_count": metadata.get("filtered_world_point_count"),
        "point_count": metadata.get("point_count"),
    }
    mismatches = {
        key: value for key, value in checks.items() if int(value or -1) != expected_count
    }
    if mismatches:
        raise KnownPoseCacheEligibilityError(
            f"cache point counts do not match S*H*W={expected_count}: {mismatches}"
        )

    if metadata.get("max_reconstruction_points") is not None:
        raise KnownPoseCacheEligibilityError("max_reconstruction_points must be null")
    if str(metadata.get("reconstruction_sample_method", "")).lower() not in {"none", "all"}:
        raise KnownPoseCacheEligibilityError("reconstruction_sample_method must be none/all")
    if float(metadata.get("min_world_point_confidence", 0.0)) > 0.0:
        raise KnownPoseCacheEligibilityError("min_world_point_confidence must be <= 0")
    if float(metadata.get("finite_world_point_ratio_before_filter", 0.0)) < 1.0:
        raise KnownPoseCacheEligibilityError("finite_world_point_ratio_before_filter must be 1.0")
    if float(metadata.get("valid_world_point_ratio", 0.0)) < 1.0:
        raise KnownPoseCacheEligibilityError("valid_world_point_ratio must be 1.0")

    return PerViewFlattenMetadata(
        view_count=view_count,
        height=height,
        width=width,
        expected_point_count=expected_count,
    )


def reshape_flattened_points_by_view(
    points: Tensor,
    metadata: dict[str, Any],
) -> tuple[Tensor, PerViewFlattenMetadata]:
    flatten = inspect_per_view_flatten_metadata(metadata)
    points = torch.as_tensor(points, dtype=torch.float32).cpu()
    if points.shape != (flatten.expected_point_count, 3):
        raise KnownPoseCacheEligibilityError(
            "points.pt shape does not match metadata: "
            f"expected {(flatten.expected_point_count, 3)}, got {tuple(points.shape)}"
        )
    if not torch.isfinite(points).all():
        raise KnownPoseCacheEligibilityError("points.pt contains NaN or Inf")
    return points.reshape(flatten.view_count, flatten.height, flatten.width, 3), flatten


def predicted_world_to_local_camera_points(
    per_view_world_points: Tensor,
    world_to_camera_extrinsics: Tensor,
) -> Tensor:
    """Convert cached VGGT world points to per-view local camera coordinates."""

    points = torch.as_tensor(per_view_world_points, dtype=torch.float32).cpu()
    extrinsics = torch.as_tensor(world_to_camera_extrinsics, dtype=torch.float32).cpu()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"per_view_world_points must have shape [S, H, W, 3], got {tuple(points.shape)}")
    if extrinsics.shape != (points.shape[0], 3, 4):
        raise ValueError(
            f"world_to_camera_extrinsics must have shape {(points.shape[0], 3, 4)}, "
            f"got {tuple(extrinsics.shape)}"
        )

    local = []
    for index in range(points.shape[0]):
        rotation = extrinsics[index, :3, :3]
        translation = extrinsics[index, :3, 3]
        flat = points[index].reshape(-1, 3)
        local.append((flat @ rotation.T + translation).reshape_as(points[index]))
    return torch.stack(local, dim=0)


def local_camera_points_to_known_world(
    per_view_local_points: Tensor,
    camera_to_world_poses: Tensor,
    depth_scale: float = 1.0,
) -> Tensor:
    """Fuse local camera points into a known camera-to-world coordinate frame."""

    points = torch.as_tensor(per_view_local_points, dtype=torch.float32).cpu()
    poses = torch.as_tensor(camera_to_world_poses, dtype=torch.float32).cpu()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"per_view_local_points must have shape [S, H, W, 3], got {tuple(points.shape)}")
    if poses.shape != (points.shape[0], 4, 4):
        raise ValueError(
            f"camera_to_world_poses must have shape {(points.shape[0], 4, 4)}, got {tuple(poses.shape)}"
        )

    fused = []
    for index in range(points.shape[0]):
        rotation = poses[index, :3, :3]
        translation = poses[index, :3, 3]
        flat = points[index].reshape(-1, 3) * float(depth_scale)
        fused.append((flat @ rotation.T + translation).reshape_as(points[index]))
    return torch.stack(fused, dim=0)


def fuse_cached_points_with_known_poses(
    points: Tensor,
    metadata: dict[str, Any],
    world_to_camera_extrinsics: Tensor,
    camera_to_world_poses: Tensor,
    depth_scale: float,
) -> tuple[Tensor, dict[str, int]]:
    """Recover local geometry from v3 cache and place it using known poses."""

    per_view_world, flatten = reshape_flattened_points_by_view(points, metadata)
    local = predicted_world_to_local_camera_points(
        per_view_world,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
    )
    fused = local_camera_points_to_known_world(
        local,
        camera_to_world_poses=camera_to_world_poses,
        depth_scale=depth_scale,
    )
    return fused.reshape(-1, 3).contiguous(), flatten.to_dict()
