"""Camera-anchored alignment for fair oracle metric evaluation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VGGT_ROOT = PROJECT_ROOT / "vggt"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

from .io import load_pose_matrix, view_id_from_path
from .metrics import SimilarityTransform, estimate_similarity_transform


@dataclass(frozen=True)
class CameraAnchorAlignment:
    """Sim(3) transform estimated from shared observed camera centers."""

    transform: SimilarityTransform
    shared_anchor_ids: list[str]
    predicted_centers: dict[str, list[float]]
    gt_centers: dict[str, list[float]]
    aligned_predicted_centers: dict[str, list[float]]
    anchor_errors: dict[str, float]
    camera_rmse: float
    camera_max_error: float
    condition_number: float
    pose_convention: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "shared_anchor_ids": self.shared_anchor_ids,
            "predicted_centers": self.predicted_centers,
            "gt_centers": self.gt_centers,
            "aligned_predicted_centers": self.aligned_predicted_centers,
            "anchor_errors": self.anchor_errors,
            "camera_rmse": self.camera_rmse,
            "camera_max_error": self.camera_max_error,
            "condition_number": self.condition_number,
            "pose_convention": self.pose_convention,
        }


def decode_vggt_pose_enc(
    pose_enc: Tensor,
    image_size_hw: tuple[int, int],
    build_intrinsics: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Decode VGGT pose encoding with the official utility."""

    if pose_enc.ndim == 2:
        pose_enc = pose_enc.unsqueeze(0)
    if pose_enc.ndim != 3 or pose_enc.shape[-1] != 9:
        raise ValueError(f"pose_enc must have shape [B, S, 9], got {tuple(pose_enc.shape)}")
    return pose_encoding_to_extri_intri(
        pose_enc.float().cpu(),
        image_size_hw=image_size_hw,
        build_intrinsics=build_intrinsics,
    )


def camera_centers_from_world_to_camera(extrinsics: Tensor) -> Tensor:
    """Return camera centers from OpenCV world-to-camera [R|t] extrinsics."""

    if extrinsics.ndim == 3:
        extrinsics = extrinsics.unsqueeze(0)
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            "extrinsics must have shape [B, S, 3, 4] or [S, 3, 4], "
            f"got {tuple(extrinsics.shape)}"
        )
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)


def camera_centers_from_camera_to_world(poses: Tensor) -> Tensor:
    """Return camera centers from camera-to-world 4x4 matrices."""

    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must have shape [N, 4, 4] or [4, 4], got {tuple(poses.shape)}")
    return poses[:, :3, 3].float().cpu()


def predicted_camera_centers_from_pose_enc(
    pose_enc: Tensor,
    image_size_hw: tuple[int, int],
) -> Tensor:
    extrinsics, _intrinsics = decode_vggt_pose_enc(
        pose_enc,
        image_size_hw=image_size_hw,
        build_intrinsics=False,
    )
    return camera_centers_from_world_to_camera(extrinsics).squeeze(0)


def load_pose_enc(path: Path) -> Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def infer_image_size_hw(metadata: dict[str, Any]) -> tuple[int, int]:
    shape = metadata.get("input_shape")
    if not isinstance(shape, list | tuple) or len(shape) < 2:
        raise ValueError("metadata input_shape is required to decode VGGT pose_enc")
    return int(shape[-2]), int(shape[-1])


def view_ids_from_image_paths(image_paths: list[str]) -> list[str]:
    return [view_id_from_path(Path(path)) for path in image_paths]


def load_reconstruction_camera_centers(reconstruction_dir: Path) -> dict[str, Tensor]:
    """Load decoded predicted camera centers keyed by view id."""

    metadata_path = reconstruction_dir / "metadata.json"
    pose_enc_path = reconstruction_dir / "pose_enc.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing reconstruction metadata: {metadata_path}")
    if not pose_enc_path.is_file():
        raise FileNotFoundError(f"Missing reconstruction pose_enc: {pose_enc_path}")

    import json

    metadata = json.loads(metadata_path.read_text())
    image_paths = metadata.get("image_paths")
    if not isinstance(image_paths, list):
        raise ValueError(f"metadata image_paths must be a list in {metadata_path}")
    pose_enc = load_pose_enc(pose_enc_path)
    centers = predicted_camera_centers_from_pose_enc(
        pose_enc,
        image_size_hw=infer_image_size_hw(metadata),
    )
    if centers.shape[0] != len(image_paths):
        raise ValueError(
            f"pose_enc view count {centers.shape[0]} does not match image_paths "
            f"count {len(image_paths)} in {metadata_path}"
        )
    return {
        view_id: centers[index].float().cpu()
        for index, view_id in enumerate(view_ids_from_image_paths(image_paths))
    }


def load_gt_camera_centers_by_view_id(view_ids: list[str], pose_dir: Path) -> dict[str, Tensor]:
    centers = {}
    for view_id in view_ids:
        pose = torch.tensor(load_pose_matrix(pose_dir / f"{view_id}.txt"), dtype=torch.float32)
        centers[view_id] = camera_centers_from_camera_to_world(pose)[0]
    return centers


def camera_anchor_condition_number(centers: Tensor) -> float:
    """Return line-vs-plane conditioning for centered anchor centers.

    With exactly three anchors, centered 3D points have rank at most two. The
    useful degeneracy check is therefore sigma_0 / sigma_1; infinity means the
    anchors are effectively collinear or coincident.
    """

    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError(f"centers must have shape [N, 3], got {tuple(centers.shape)}")
    if centers.shape[0] < 3:
        return math.inf
    centered = centers.float().cpu() - centers.float().cpu().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    if singular_values.numel() < 2 or singular_values[1].abs().item() < 1e-12:
        return math.inf
    return (singular_values[0] / singular_values[1]).item()


def estimate_camera_anchor_alignment(
    predicted_centers_by_view: dict[str, Tensor],
    gt_centers_by_view: dict[str, Tensor],
    shared_anchor_ids: list[str],
    pose_convention: str = "vggt_world_to_camera__scannet_camera_to_world",
) -> CameraAnchorAlignment:
    missing_pred = [view_id for view_id in shared_anchor_ids if view_id not in predicted_centers_by_view]
    missing_gt = [view_id for view_id in shared_anchor_ids if view_id not in gt_centers_by_view]
    if missing_pred or missing_gt:
        raise KeyError(f"Missing predicted anchors {missing_pred} or GT anchors {missing_gt}")
    if len(shared_anchor_ids) < 3:
        raise ValueError("At least three shared camera anchors are required for Sim(3)")

    predicted = torch.stack([predicted_centers_by_view[view_id] for view_id in shared_anchor_ids]).float().cpu()
    gt = torch.stack([gt_centers_by_view[view_id] for view_id in shared_anchor_ids]).float().cpu()
    transform = estimate_similarity_transform(predicted, gt)
    aligned = transform.apply(predicted)
    errors = torch.linalg.norm(aligned - gt, dim=1)
    return CameraAnchorAlignment(
        transform=transform,
        shared_anchor_ids=list(shared_anchor_ids),
        predicted_centers={
            view_id: predicted[index].tolist() for index, view_id in enumerate(shared_anchor_ids)
        },
        gt_centers={
            view_id: gt[index].tolist() for index, view_id in enumerate(shared_anchor_ids)
        },
        aligned_predicted_centers={
            view_id: aligned[index].tolist() for index, view_id in enumerate(shared_anchor_ids)
        },
        anchor_errors={
            view_id: errors[index].item() for index, view_id in enumerate(shared_anchor_ids)
        },
        camera_rmse=errors.square().mean().sqrt().item(),
        camera_max_error=errors.max().item(),
        condition_number=camera_anchor_condition_number(predicted),
        pose_convention=pose_convention,
    )
