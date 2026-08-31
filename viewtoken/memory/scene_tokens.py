"""Construct world-aligned scene tokens from VGGT patch evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SceneTokenBatch:
    """Patch features paired with confidence-weighted world positions.

    Tensors retain their `[B, S, N, ...]` axes. Phase 0 intentionally keeps
    tokens from different observed views separate instead of prematurely
    merging features that may be view dependent.
    """

    features: Tensor
    positions: Tensor
    confidence: Tensor
    valid_mask: Tensor

    def flatten_valid(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return valid features, positions, and confidence as flat tensors."""

        mask = self.valid_mask.reshape(-1)
        feature_dim = self.features.shape[-1]
        return (
            self.features.reshape(-1, feature_dim)[mask],
            self.positions.reshape(-1, 3)[mask],
            self.confidence.reshape(-1)[mask],
        )


def _validate_inputs(
    patch_tokens: Tensor,
    world_points: Tensor,
    world_points_conf: Tensor,
    patch_grid: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    if patch_tokens.ndim != 4:
        raise ValueError(
            f"patch_tokens must have shape [B, S, N, C], got {patch_tokens.shape}"
        )
    if world_points.ndim != 5 or world_points.shape[-1] != 3:
        raise ValueError(
            "world_points must have shape [B, S, H, W, 3], "
            f"got {world_points.shape}"
        )
    if world_points_conf.ndim == 5 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf.squeeze(-1)
    if world_points_conf.ndim != 4:
        raise ValueError(
            "world_points_conf must have shape [B, S, H, W], "
            f"got {world_points_conf.shape}"
        )

    batch, views, token_count, _feature_dim = patch_tokens.shape
    point_batch, point_views, height, width, _xyz = world_points.shape
    if (batch, views) != (point_batch, point_views):
        raise ValueError("Patch tokens and world points have different B/S dimensions")
    if world_points_conf.shape != (batch, views, height, width):
        raise ValueError("World-point confidence does not match world-point dimensions")

    grid_height, grid_width = patch_grid
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"Invalid patch grid: {patch_grid}")
    if token_count != grid_height * grid_width:
        raise ValueError(
            f"Token count {token_count} does not match patch grid {patch_grid}"
        )
    if height % grid_height or width % grid_width:
        raise ValueError(
            f"Image grid {(height, width)} is not divisible by patch grid {patch_grid}"
        )
    return batch, views, height, width, grid_height, grid_width


def build_patch_scene_tokens(
    patch_tokens: Tensor,
    world_points: Tensor,
    world_points_conf: Tensor,
    patch_grid: tuple[int, int],
    min_confidence: float = 0.0,
    eps: float = 1e-8,
) -> SceneTokenBatch:
    """Lift VGGT patch tokens into world coordinates.

    Every patch receives the confidence-weighted centroid of the dense VGGT
    world points in the corresponding image region. Non-finite points and
    pixels at or below `min_confidence` are ignored. The returned patch order is
    row-major and therefore matches the patch embedding token order.
    """

    if world_points_conf.ndim == 5 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf.squeeze(-1)

    batch, views, height, width, grid_height, grid_width = _validate_inputs(
        patch_tokens, world_points, world_points_conf, patch_grid
    )
    patch_height = height // grid_height
    patch_width = width // grid_width
    pixels_per_patch = patch_height * patch_width

    points = (
        world_points.reshape(
            batch, views, grid_height, patch_height, grid_width, patch_width, 3
        )
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(batch, views, grid_height * grid_width, pixels_per_patch, 3)
    )
    confidence = (
        world_points_conf.reshape(
            batch, views, grid_height, patch_height, grid_width, patch_width
        )
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, views, grid_height * grid_width, pixels_per_patch)
    )

    finite_points = torch.isfinite(points).all(dim=-1)
    finite_confidence = torch.isfinite(confidence)
    pixel_valid = finite_points & finite_confidence & (confidence > min_confidence)
    weights = torch.where(pixel_valid, confidence.clamp_min(0), 0)

    weight_sum = weights.sum(dim=-1)
    valid_count = pixel_valid.sum(dim=-1)
    patch_valid = (weight_sum > eps) & (valid_count > 0)

    safe_points = torch.where(finite_points.unsqueeze(-1), points, 0)
    weighted_position = (safe_points * weights.unsqueeze(-1)).sum(dim=-2)
    positions = weighted_position / weight_sum.clamp_min(eps).unsqueeze(-1)
    positions = torch.where(patch_valid.unsqueeze(-1), positions, 0)

    patch_confidence = weight_sum / valid_count.clamp_min(1)
    patch_confidence = torch.where(patch_valid, patch_confidence, 0)

    return SceneTokenBatch(
        features=patch_tokens,
        positions=positions,
        confidence=patch_confidence,
        valid_mask=patch_valid,
    )

