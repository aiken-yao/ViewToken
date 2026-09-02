"""GT surface visibility helpers for offline oracle-gain audits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole camera intrinsics tied to an image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_matrix(
        cls,
        matrix: Tensor,
        width: int,
        height: int,
    ) -> "PinholeIntrinsics":
        matrix = torch.as_tensor(matrix, dtype=torch.float64)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(f"intrinsics matrix must be at least 3x3, got {tuple(matrix.shape)}")
        return cls(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            width=int(width),
            height=int(height),
        )

    def scaled_to(self, width: int, height: int) -> "PinholeIntrinsics":
        scale_x = float(width) / float(self.width)
        scale_y = float(height) / float(self.height)
        return PinholeIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=int(width),
            height=int(height),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Projection state for world points under a ScanNet camera-to-world pose."""

    camera_points: Tensor
    uv: Tensor
    pixel_xy: Tensor
    depths: Tensor
    front_mask: Tensor
    in_frame_mask: Tensor


@dataclass(frozen=True)
class VisibilityMasks:
    observed: Tensor
    candidate: Tensor
    overlap: Tensor
    novel: Tensor
    union: Tensor


@dataclass(frozen=True)
class VisibilityMaskStats:
    total_surface_count: int
    observed_count: int
    candidate_count: int
    overlap_count: int
    novel_count: int
    union_count: int
    observed_fraction: float
    candidate_fraction: float
    overlap_fraction: float
    novel_scene_fraction: float
    union_fraction: float
    candidate_overlap_fraction: float | None
    candidate_novel_fraction: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total_surface_count": self.total_surface_count,
            "observed_count": self.observed_count,
            "candidate_count": self.candidate_count,
            "overlap_count": self.overlap_count,
            "novel_count": self.novel_count,
            "union_count": self.union_count,
            "observed_fraction": self.observed_fraction,
            "candidate_fraction": self.candidate_fraction,
            "overlap_fraction": self.overlap_fraction,
            "novel_scene_fraction": self.novel_scene_fraction,
            "union_fraction": self.union_fraction,
            "candidate_overlap_fraction": self.candidate_overlap_fraction,
            "candidate_novel_fraction": self.candidate_novel_fraction,
        }


@dataclass(frozen=True)
class CameraPoseDelta:
    min_distance_to_observed_meters: float
    nearest_observed_view_id: str | None
    min_view_direction_change_degrees: float
    nearest_view_direction_view_id: str | None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "min_distance_to_observed_meters": self.min_distance_to_observed_meters,
            "nearest_observed_view_id": self.nearest_observed_view_id,
            "min_view_direction_change_degrees": self.min_view_direction_change_degrees,
            "nearest_view_direction_view_id": self.nearest_view_direction_view_id,
        }


def _validate_points(points: Tensor) -> Tensor:
    points = torch.as_tensor(points, dtype=torch.float32).cpu()
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    return points


def _validate_pose(camera_to_world: Tensor) -> Tensor:
    pose = torch.as_tensor(camera_to_world, dtype=torch.float32).cpu()
    if pose.shape != (4, 4):
        raise ValueError(f"camera_to_world must have shape [4, 4], got {tuple(pose.shape)}")
    return pose


def transform_world_to_camera(points: Tensor, camera_to_world: Tensor) -> Tensor:
    """Transform row-vector world points into OpenCV camera coordinates.

    ScanNet pose files are treated as camera-to-world matrices. For row vectors,
    x_cam = (x_world - t_world) @ R_c2w.
    """

    points = _validate_points(points)
    pose = _validate_pose(camera_to_world)
    rotation_c2w = pose[:3, :3]
    translation = pose[:3, 3]
    return (points - translation) @ rotation_c2w


def project_world_points(
    points: Tensor,
    camera_to_world: Tensor,
    intrinsics: PinholeIntrinsics,
    near: float = 1e-5,
) -> ProjectionResult:
    """Project world points using ScanNet c2w poses and OpenCV +Z camera depth."""

    points = _validate_points(points)
    camera_points = transform_world_to_camera(points, camera_to_world)
    depths = camera_points[:, 2]
    finite = torch.isfinite(points).all(dim=-1) & torch.isfinite(camera_points).all(dim=-1)
    front = finite & (depths > float(near))

    uv = torch.full((points.shape[0], 2), float("nan"), dtype=torch.float32)
    if front.any():
        front_points = camera_points[front]
        front_depths = front_points[:, 2].clamp_min(float(near))
        uv_front = torch.empty((front_points.shape[0], 2), dtype=torch.float32)
        uv_front[:, 0] = intrinsics.fx * front_points[:, 0] / front_depths + intrinsics.cx
        uv_front[:, 1] = intrinsics.fy * front_points[:, 1] / front_depths + intrinsics.cy
        uv[front] = uv_front

    in_frame = (
        front
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(intrinsics.width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(intrinsics.height))
    )
    pixel_xy = torch.full((points.shape[0], 2), -1, dtype=torch.long)
    if in_frame.any():
        pixel_xy[in_frame, 0] = torch.floor(uv[in_frame, 0]).to(torch.long)
        pixel_xy[in_frame, 1] = torch.floor(uv[in_frame, 1]).to(torch.long)

    return ProjectionResult(
        camera_points=camera_points,
        uv=uv,
        pixel_xy=pixel_xy,
        depths=depths,
        front_mask=front,
        in_frame_mask=in_frame,
    )


def _minimum_depth_per_pixel(
    linear_pixels: Tensor,
    depths: Tensor,
    pixel_count: int,
) -> Tensor:
    depth_buffer = torch.full((pixel_count,), float("inf"), dtype=torch.float32)
    if linear_pixels.numel() == 0:
        return depth_buffer
    if hasattr(depth_buffer, "scatter_reduce_"):
        depth_buffer.scatter_reduce_(
            0,
            linear_pixels.to(torch.long),
            depths.to(torch.float32),
            reduce="amin",
            include_self=True,
        )
        return depth_buffer

    order = torch.argsort(linear_pixels)
    sorted_pixels = linear_pixels[order]
    sorted_depths = depths[order]
    unique_pixels, inverse = torch.unique_consecutive(sorted_pixels, return_inverse=True)
    mins = torch.full((unique_pixels.shape[0],), float("inf"), dtype=torch.float32)
    for index in range(sorted_depths.shape[0]):
        slot = int(inverse[index].item())
        value = float(sorted_depths[index].item())
        if value < float(mins[slot].item()):
            mins[slot] = value
    depth_buffer[unique_pixels] = mins
    return depth_buffer


def visible_surface_mask(
    points: Tensor,
    camera_to_world: Tensor,
    intrinsics: PinholeIntrinsics,
    depth_tolerance: float,
    near: float = 1e-5,
    pixel_radius: int = 0,
) -> Tensor:
    """Return points visible from one camera after z-buffer occlusion."""

    points = _validate_points(points)
    pixel_radius = int(pixel_radius)
    if pixel_radius < 0:
        raise ValueError("pixel_radius must be non-negative")
    projection = project_world_points(
        points,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        near=near,
    )
    visible = torch.zeros((points.shape[0],), dtype=torch.bool)
    candidate_indices = torch.nonzero(projection.in_frame_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return visible

    pixel_xy = projection.pixel_xy[candidate_indices]
    depths = projection.depths[candidate_indices].to(torch.float32)
    if pixel_radius == 0:
        linear_pixels = pixel_xy[:, 1] * int(intrinsics.width) + pixel_xy[:, 0]
        depth_buffer = _minimum_depth_per_pixel(
            linear_pixels=linear_pixels,
            depths=depths,
            pixel_count=int(intrinsics.width) * int(intrinsics.height),
        )
        nearest_depths = depth_buffer[linear_pixels]
        visible[candidate_indices] = depths <= nearest_depths + float(depth_tolerance)
        return visible

    offsets = torch.tensor(
        [
            (dx, dy)
            for dy in range(-pixel_radius, pixel_radius + 1)
            for dx in range(-pixel_radius, pixel_radius + 1)
        ],
        dtype=torch.long,
    )
    expanded_xy = pixel_xy[:, None, :] + offsets[None, :, :]
    inside = (
        (expanded_xy[..., 0] >= 0)
        & (expanded_xy[..., 0] < int(intrinsics.width))
        & (expanded_xy[..., 1] >= 0)
        & (expanded_xy[..., 1] < int(intrinsics.height))
    )
    expanded_point_positions = (
        torch.arange(candidate_indices.shape[0], dtype=torch.long)[:, None]
        .expand(-1, offsets.shape[0])
    )[inside]
    expanded_depths = depths[expanded_point_positions]
    expanded_xy = expanded_xy[inside]
    linear_pixels = expanded_xy[:, 1] * int(intrinsics.width) + expanded_xy[:, 0]
    depth_buffer = _minimum_depth_per_pixel(
        linear_pixels=linear_pixels,
        depths=expanded_depths,
        pixel_count=int(intrinsics.width) * int(intrinsics.height),
    )
    nearest_depths = depth_buffer[linear_pixels]
    expanded_visible = expanded_depths <= nearest_depths + float(depth_tolerance)
    per_candidate_visible = torch.zeros((candidate_indices.shape[0],), dtype=torch.int32)
    per_candidate_visible.index_add_(
        0,
        expanded_point_positions,
        expanded_visible.to(torch.int32),
    )
    visible[candidate_indices] = per_candidate_visible > 0
    return visible


def union_visible_surface_mask(
    points: Tensor,
    camera_to_world_poses: Iterable[Tensor],
    intrinsics: PinholeIntrinsics,
    depth_tolerance: float,
    near: float = 1e-5,
    pixel_radius: int = 0,
) -> Tensor:
    """Return the union of per-camera visible surface masks."""

    points = _validate_points(points)
    union = torch.zeros((points.shape[0],), dtype=torch.bool)
    for pose in camera_to_world_poses:
        union |= visible_surface_mask(
            points,
            camera_to_world=pose,
            intrinsics=intrinsics,
            depth_tolerance=depth_tolerance,
            near=near,
            pixel_radius=pixel_radius,
        )
    return union


def build_visibility_masks(
    points: Tensor,
    observed_camera_to_world_poses: Iterable[Tensor],
    candidate_camera_to_world: Tensor,
    intrinsics: PinholeIntrinsics,
    depth_tolerance: float,
    near: float = 1e-5,
    pixel_radius: int = 0,
) -> VisibilityMasks:
    observed = union_visible_surface_mask(
        points,
        observed_camera_to_world_poses,
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        near=near,
        pixel_radius=pixel_radius,
    )
    candidate = visible_surface_mask(
        points,
        camera_to_world=candidate_camera_to_world,
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        near=near,
        pixel_radius=pixel_radius,
    )
    overlap = observed & candidate
    novel = candidate & ~observed
    union = observed | candidate
    return VisibilityMasks(
        observed=observed,
        candidate=candidate,
        overlap=overlap,
        novel=novel,
        union=union,
    )


def summarize_visibility_masks(masks: VisibilityMasks) -> VisibilityMaskStats:
    total_count = int(masks.observed.numel())
    if total_count == 0:
        raise ValueError("visibility masks must be non-empty")
    observed_count = int(masks.observed.sum().item())
    candidate_count = int(masks.candidate.sum().item())
    overlap_count = int(masks.overlap.sum().item())
    novel_count = int(masks.novel.sum().item())
    union_count = int(masks.union.sum().item())

    def scene_fraction(count: int) -> float:
        return float(count) / float(total_count)

    return VisibilityMaskStats(
        total_surface_count=total_count,
        observed_count=observed_count,
        candidate_count=candidate_count,
        overlap_count=overlap_count,
        novel_count=novel_count,
        union_count=union_count,
        observed_fraction=scene_fraction(observed_count),
        candidate_fraction=scene_fraction(candidate_count),
        overlap_fraction=scene_fraction(overlap_count),
        novel_scene_fraction=scene_fraction(novel_count),
        union_fraction=scene_fraction(union_count),
        candidate_overlap_fraction=(
            None if candidate_count == 0 else float(overlap_count) / float(candidate_count)
        ),
        candidate_novel_fraction=(
            None if candidate_count == 0 else float(novel_count) / float(candidate_count)
        ),
    )


def camera_center_from_pose(camera_to_world: Tensor) -> Tensor:
    pose = _validate_pose(camera_to_world)
    return pose[:3, 3].float().cpu()


def camera_forward_from_pose(camera_to_world: Tensor) -> Tensor:
    """Return the OpenCV +Z viewing direction in world coordinates."""

    pose = _validate_pose(camera_to_world)
    forward = pose[:3, 2].float().cpu()
    return forward / torch.linalg.norm(forward).clamp_min(1e-12)


def camera_pose_delta_to_observed(
    candidate_camera_to_world: Tensor,
    observed_camera_to_world_poses: dict[str, Tensor],
) -> CameraPoseDelta:
    if not observed_camera_to_world_poses:
        return CameraPoseDelta(
            min_distance_to_observed_meters=math.inf,
            nearest_observed_view_id=None,
            min_view_direction_change_degrees=math.inf,
            nearest_view_direction_view_id=None,
        )
    candidate_center = camera_center_from_pose(candidate_camera_to_world)
    candidate_forward = camera_forward_from_pose(candidate_camera_to_world)
    nearest_distance = math.inf
    nearest_distance_id: str | None = None
    nearest_angle = math.inf
    nearest_angle_id: str | None = None
    for view_id, pose in observed_camera_to_world_poses.items():
        observed_center = camera_center_from_pose(pose)
        distance = torch.linalg.norm(candidate_center - observed_center).item()
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_distance_id = view_id
        observed_forward = camera_forward_from_pose(pose)
        cosine = torch.dot(candidate_forward, observed_forward).item()
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle < nearest_angle:
            nearest_angle = angle
            nearest_angle_id = view_id
    return CameraPoseDelta(
        min_distance_to_observed_meters=nearest_distance,
        nearest_observed_view_id=nearest_distance_id,
        min_view_direction_change_degrees=nearest_angle,
        nearest_view_direction_view_id=nearest_angle_id,
    )
