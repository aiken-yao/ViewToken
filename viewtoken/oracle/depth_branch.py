"""Depth backprojection and preprocessing-intrinsics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class ImagePreprocessTransform:
    mode: str
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    crop_left: int = 0
    crop_top: int = 0
    pad_left: int = 0
    pad_top: int = 0
    batch_pad_left: int = 0
    batch_pad_top: int = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mode": self.mode,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "crop_left": self.crop_left,
            "crop_top": self.crop_top,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "batch_pad_left": self.batch_pad_left,
            "batch_pad_top": self.batch_pad_top,
        }


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return int(round(float(value) / float(patch_size)) * int(patch_size))


def compute_image_preprocess_transform(
    width: int,
    height: int,
    mode: str = "crop",
    target_size: int = 518,
    patch_size: int = 14,
) -> ImagePreprocessTransform:
    """Mirror VGGT load_and_preprocess_images resize/crop/pad geometry."""

    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be crop or pad")
    width = int(width)
    height = int(height)
    target_size = int(target_size)
    patch_size = int(patch_size)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    if mode == "crop":
        resized_width = target_size
        resized_height = _round_to_patch_multiple(height * (resized_width / width), patch_size)
        crop_top = max(0, (resized_height - target_size) // 2)
        output_height = target_size if resized_height > target_size else resized_height
        return ImagePreprocessTransform(
            mode=mode,
            original_width=width,
            original_height=height,
            resized_width=resized_width,
            resized_height=resized_height,
            output_width=resized_width,
            output_height=output_height,
            scale_x=resized_width / width,
            scale_y=resized_height / height,
            crop_top=crop_top,
        )

    if width >= height:
        resized_width = target_size
        resized_height = _round_to_patch_multiple(height * (resized_width / width), patch_size)
    else:
        resized_height = target_size
        resized_width = _round_to_patch_multiple(width * (resized_height / height), patch_size)
    pad_top = max(0, (target_size - resized_height) // 2)
    pad_left = max(0, (target_size - resized_width) // 2)
    return ImagePreprocessTransform(
        mode=mode,
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        output_width=target_size,
        output_height=target_size,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def compute_batch_preprocess_transforms(
    image_sizes_wh: Iterable[tuple[int, int]],
    mode: str = "crop",
    target_size: int = 518,
    patch_size: int = 14,
) -> list[ImagePreprocessTransform]:
    """Return transforms including VGGT's final centered batch padding."""

    transforms = [
        compute_image_preprocess_transform(
            width=width,
            height=height,
            mode=mode,
            target_size=target_size,
            patch_size=patch_size,
        )
        for width, height in image_sizes_wh
    ]
    if not transforms:
        raise ValueError("at least one image size is required")
    max_width = max(transform.output_width for transform in transforms)
    max_height = max(transform.output_height for transform in transforms)
    padded = []
    for transform in transforms:
        batch_pad_left = max(0, (max_width - transform.output_width) // 2)
        batch_pad_top = max(0, (max_height - transform.output_height) // 2)
        padded.append(
            ImagePreprocessTransform(
                mode=transform.mode,
                original_width=transform.original_width,
                original_height=transform.original_height,
                resized_width=transform.resized_width,
                resized_height=transform.resized_height,
                output_width=max_width,
                output_height=max_height,
                scale_x=transform.scale_x,
                scale_y=transform.scale_y,
                crop_left=transform.crop_left,
                crop_top=transform.crop_top,
                pad_left=transform.pad_left,
                pad_top=transform.pad_top,
                batch_pad_left=batch_pad_left,
                batch_pad_top=batch_pad_top,
            )
        )
    return padded


def build_per_view_shape_offsets(
    view_ids: Iterable[str],
    height: int,
    width: int,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    offset = 0
    point_count = int(height) * int(width)
    for view_id in view_ids:
        rows.append(
            {
                "view_id": str(view_id),
                "height": int(height),
                "width": int(width),
                "point_offset": offset,
                "point_count": point_count,
            }
        )
        offset += point_count
    return rows


def save_v4_depth_artifacts(
    output_dir: Path | str,
    depth: Tensor,
    depth_conf: Tensor,
    predicted_intrinsics: Tensor,
    transformed_gt_intrinsics: Tensor,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "depth": output_path / "depth.pt",
        "depth_conf": output_path / "depth_conf.pt",
        "predicted_intrinsics": output_path / "predicted_intrinsics.pt",
        "transformed_gt_intrinsics": output_path / "transformed_gt_intrinsics.pt",
    }
    torch.save(torch.as_tensor(depth).detach().float().cpu().contiguous(), artifacts["depth"])
    torch.save(torch.as_tensor(depth_conf).detach().float().cpu().contiguous(), artifacts["depth_conf"])
    torch.save(
        torch.as_tensor(predicted_intrinsics).detach().float().cpu().contiguous(),
        artifacts["predicted_intrinsics"],
    )
    torch.save(
        torch.as_tensor(transformed_gt_intrinsics).detach().float().cpu().contiguous(),
        artifacts["transformed_gt_intrinsics"],
    )
    return {key: str(path) for key, path in artifacts.items()}


def transform_intrinsics(
    intrinsics: Tensor,
    transform: ImagePreprocessTransform,
) -> Tensor:
    """Transform original pixel intrinsics into the preprocessed image frame."""

    matrix = torch.as_tensor(intrinsics, dtype=torch.float32).clone()
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3, 3], got {tuple(matrix.shape)}")
    matrix[0, 0] *= float(transform.scale_x)
    matrix[1, 1] *= float(transform.scale_y)
    matrix[0, 2] = (
        matrix[0, 2] * float(transform.scale_x)
        - float(transform.crop_left)
        + float(transform.pad_left)
        + float(transform.batch_pad_left)
    )
    matrix[1, 2] = (
        matrix[1, 2] * float(transform.scale_y)
        - float(transform.crop_top)
        + float(transform.pad_top)
        + float(transform.batch_pad_top)
    )
    matrix[0, 1] *= float(transform.scale_x)
    matrix[1, 0] *= float(transform.scale_y)
    return matrix


def depth_to_local_camera_points(depth: Tensor, intrinsics: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Backproject depth to OpenCV camera coordinates using VGGT geometry math."""

    depth = torch.as_tensor(depth, dtype=torch.float32).cpu()
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32).cpu()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H, W] or [H, W, 1], got {tuple(depth.shape)}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3, 3], got {tuple(intrinsics.shape)}")
    height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    fx = intrinsics[0, 0].clamp_min(eps)
    fy = intrinsics[1, 1].clamp_min(eps)
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    local = torch.stack(
        [
            (x - cx) * depth / fx,
            (y - cy) * depth / fy,
            depth,
        ],
        dim=-1,
    )
    return local, depth > float(eps)


def local_camera_points_to_world(local_points: Tensor, camera_to_world: Tensor) -> Tensor:
    points = torch.as_tensor(local_points, dtype=torch.float32).cpu()
    pose = torch.as_tensor(camera_to_world, dtype=torch.float32).cpu()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"local_points must have shape [H, W, 3], got {tuple(points.shape)}")
    if pose.shape != (4, 4):
        raise ValueError(f"camera_to_world must have shape [4, 4], got {tuple(pose.shape)}")
    flat = points.reshape(-1, 3)
    world = flat @ pose[:3, :3].T + pose[:3, 3]
    return world.reshape_as(points)


def depth_to_known_world_points(
    depth: Tensor,
    intrinsics: Tensor,
    camera_to_world: Tensor,
) -> tuple[Tensor, Tensor]:
    local, valid = depth_to_local_camera_points(depth, intrinsics)
    return local_camera_points_to_world(local, camera_to_world), valid


def depth_views_to_known_world_points(
    depth: Tensor,
    intrinsics: Tensor,
    camera_to_world_poses: Tensor,
    depth_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    depth = torch.as_tensor(depth, dtype=torch.float32).cpu()
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32).cpu()
    poses = torch.as_tensor(camera_to_world_poses, dtype=torch.float32).cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape [S, H, W] or [S, H, W, 1], got {tuple(depth.shape)}")
    if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics[0]
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError(
            f"intrinsics must have shape {(depth.shape[0], 3, 3)} or [1, S, 3, 3], "
            f"got {tuple(intrinsics.shape)}"
        )
    if poses.shape != (depth.shape[0], 4, 4):
        raise ValueError(f"camera_to_world_poses must have shape {(depth.shape[0], 4, 4)}, got {tuple(poses.shape)}")

    worlds = []
    valids = []
    for index in range(depth.shape[0]):
        local, valid = depth_to_local_camera_points(depth[index], intrinsics[index])
        local = local * float(depth_scale)
        worlds.append(local_camera_points_to_world(local, poses[index]))
        valids.append(valid)
    return torch.stack(worlds, dim=0), torch.stack(valids, dim=0)
