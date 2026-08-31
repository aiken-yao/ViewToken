from __future__ import annotations

import unittest

import torch

from viewtoken.memory import build_patch_scene_tokens


class SceneTokenBuilderTest(unittest.TestCase):
    def test_patch_positions_follow_row_major_token_order(self) -> None:
        patch_tokens = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        world_points = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()
        world_points = world_points.reshape(1, 1, 4, 4, 3)
        confidence = torch.ones(1, 1, 4, 4)

        scene = build_patch_scene_tokens(
            patch_tokens, world_points, confidence, patch_grid=(2, 2)
        )

        expected_positions = torch.tensor(
            [[[[0.5, 0.5, 1.0], [2.5, 0.5, 1.0],
               [0.5, 2.5, 1.0], [2.5, 2.5, 1.0]]]]
        )
        self.assertTrue(torch.allclose(scene.positions, expected_positions))
        self.assertTrue(scene.valid_mask.all())
        self.assertTrue(torch.equal(scene.features, patch_tokens))

    def test_invalid_pixels_do_not_pollute_patch_centroid(self) -> None:
        patch_tokens = torch.zeros(1, 1, 1, 2)
        world_points = torch.tensor(
            [[[[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[float("nan"), 0.0, 0.0], [9.0, 0.0, 0.0]]]]]
        )
        confidence = torch.tensor([[[[1.0, 3.0], [5.0, 0.0]]]])

        scene = build_patch_scene_tokens(
            patch_tokens, world_points, confidence, patch_grid=(1, 1)
        )

        self.assertTrue(torch.allclose(scene.positions[0, 0, 0], torch.tensor([2.5, 0, 0])))
        self.assertAlmostEqual(scene.confidence.item(), 2.0)
        self.assertTrue(scene.valid_mask.item())


if __name__ == "__main__":
    unittest.main()
