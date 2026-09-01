from __future__ import annotations

import unittest

import torch
from torch import nn

from viewtoken.backbones import VGGTFeatureExtractor


class FakeAggregator(nn.Module):
    patch_size = 14

    def forward(self, images: torch.Tensor):
        batch, views, _channels, height, width = images.shape
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        patch_start_idx = 5
        outputs = [None] * 24
        outputs[23] = torch.randn(batch, views, patch_start_idx + patch_count, 8)
        return outputs, patch_start_idx


class FakeVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = FakeAggregator()

    def forward(self, images: torch.Tensor):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, views, _channels, height, width = images.shape
        self.aggregator(images)
        return {
            "depth": torch.ones(batch, views, height, width, 1),
            "depth_conf": torch.ones(batch, views, height, width),
            "world_points": torch.ones(batch, views, height, width, 3),
            "world_points_conf": torch.ones(batch, views, height, width),
            "pose_enc": torch.ones(batch, views, 9),
        }


class VGGTFeatureExtractorTest(unittest.TestCase):
    def test_extract_removes_special_tokens_and_keeps_geometry(self) -> None:
        images = torch.rand(2, 3, 28, 28)
        features = VGGTFeatureExtractor(FakeVGGT(), layer_index=23).extract(images)

        self.assertEqual(features.patch_tokens.shape, (1, 2, 4, 8))
        self.assertEqual(features.patch_grid, (2, 2))
        self.assertEqual(features.patch_start_idx, 5)
        self.assertEqual(features.aggregator_forward_count, 1)
        self.assertEqual(features.depth.shape, (1, 2, 28, 28, 1))
        self.assertEqual(features.world_points.shape, (1, 2, 28, 28, 3))
        self.assertIsNotNone(features.pose_enc)

    def test_extract_rejects_uncached_layer(self) -> None:
        images = torch.rand(2, 3, 28, 28)
        extractor = VGGTFeatureExtractor(FakeVGGT(), layer_index=22)

        with self.assertRaisesRegex(ValueError, "not cached"):
            extractor.extract(images)


if __name__ == "__main__":
    unittest.main()
