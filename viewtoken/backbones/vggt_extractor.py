"""Extract patch tokens and geometric predictions from a frozen VGGT model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VGGTFeatureBatch:
    """Outputs needed to construct the first ViewToken scene memory."""

    patch_tokens: Tensor
    depth: Tensor
    depth_conf: Tensor
    world_points: Tensor
    world_points_conf: Tensor
    pose_enc: Tensor | None
    patch_start_idx: int
    patch_grid: tuple[int, int]
    layer_index: int

    def tensor_dict(self) -> dict[str, Tensor]:
        tensors = {
            "patch_tokens": self.patch_tokens,
            "depth": self.depth,
            "depth_conf": self.depth_conf,
            "world_points": self.world_points,
            "world_points_conf": self.world_points_conf,
        }
        if self.pose_enc is not None:
            tensors["pose_enc"] = self.pose_enc
        return tensors


class VGGTFeatureExtractor:
    """Single-pass adapter around the official VGGT implementation.

    VGGT's public forward method does not return aggregator tokens. A temporary
    forward hook captures the aggregator output while the normal heads compute
    depth, confidence, world points, and pose. This avoids a second expensive
    aggregator pass and leaves the upstream VGGT source unchanged.
    """

    REQUIRED_PREDICTIONS = (
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
    )

    def __init__(self, model: nn.Module, layer_index: int = 23) -> None:
        if not hasattr(model, "aggregator"):
            raise TypeError("VGGT model must expose an 'aggregator' module")

        self.model = model
        self.layer_index = layer_index

    @staticmethod
    def _resolve_layer(
        aggregated_tokens: list[Tensor | None], layer_index: int
    ) -> tuple[Tensor, int]:
        resolved_index = layer_index
        if resolved_index < 0:
            resolved_index += len(aggregated_tokens)

        if not 0 <= resolved_index < len(aggregated_tokens):
            raise IndexError(
                f"Layer {layer_index} is outside aggregator output range "
                f"[0, {len(aggregated_tokens) - 1}]"
            )

        layer_tokens = aggregated_tokens[resolved_index]
        if layer_tokens is None:
            cached = [
                index for index, tokens in enumerate(aggregated_tokens) if tokens is not None
            ]
            raise ValueError(
                f"Aggregator layer {resolved_index} is not cached; available layers: {cached}"
            )
        if layer_tokens.ndim != 4:
            raise ValueError(
                "Expected aggregator tokens with shape [B, S, P, C], "
                f"received {tuple(layer_tokens.shape)}"
            )
        return layer_tokens, resolved_index

    @torch.inference_mode()
    def extract(self, images: Tensor) -> VGGTFeatureBatch:
        """Run VGGT once and return patch-level appearance and geometry.

        Args:
            images: Tensor shaped `[S, 3, H, W]` or `[B, S, 3, H, W]`,
                with values in `[0, 1]`.
        """

        if images.ndim not in (4, 5):
            raise ValueError(
                "Images must have shape [S, 3, H, W] or [B, S, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        captured: dict[str, Any] = {}

        def capture_aggregator_output(
            _module: nn.Module, _args: tuple[Any, ...], output: Any
        ) -> None:
            captured["aggregator_output"] = output

        handle = self.model.aggregator.register_forward_hook(capture_aggregator_output)
        try:
            predictions = self.model(images)
        finally:
            handle.remove()

        if "aggregator_output" not in captured:
            raise RuntimeError("VGGT aggregator hook did not capture an output")

        aggregated_tokens, patch_start_idx = captured["aggregator_output"]
        layer_tokens, resolved_layer = self._resolve_layer(
            aggregated_tokens, self.layer_index
        )
        patch_tokens = layer_tokens[:, :, patch_start_idx:, :]

        height, width = images.shape[-2:]
        patch_size = int(getattr(self.model.aggregator, "patch_size", 14))
        patch_grid = (height // patch_size, width // patch_size)
        expected_patch_count = patch_grid[0] * patch_grid[1]
        if patch_tokens.shape[2] != expected_patch_count:
            raise ValueError(
                "Patch-token count does not match the input grid: "
                f"got {patch_tokens.shape[2]}, expected {expected_patch_count} "
                f"for grid {patch_grid} and patch size {patch_size}"
            )

        missing = [key for key in self.REQUIRED_PREDICTIONS if key not in predictions]
        if missing:
            raise KeyError(
                "VGGT must enable its depth and point heads; missing predictions: "
                + ", ".join(missing)
            )

        return VGGTFeatureBatch(
            patch_tokens=patch_tokens,
            depth=predictions["depth"],
            depth_conf=predictions["depth_conf"],
            world_points=predictions["world_points"],
            world_points_conf=predictions["world_points_conf"],
            pose_enc=predictions.get("pose_enc"),
            patch_start_idx=int(patch_start_idx),
            patch_grid=patch_grid,
            layer_index=resolved_layer,
        )

