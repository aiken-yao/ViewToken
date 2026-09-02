from __future__ import annotations

import unittest

import torch

from viewtoken.oracle import (
    build_per_view_shape_offsets,
    compute_batch_preprocess_transforms,
    compute_image_preprocess_transform,
    depth_to_known_world_points,
    depth_to_local_camera_points,
    depth_views_to_known_world_points,
    transform_intrinsics,
)


class DepthBranchTest(unittest.TestCase):
    def test_scannet_crop_transform_matches_vggt_518_width_path(self) -> None:
        transform = compute_image_preprocess_transform(1296, 968, mode="crop", target_size=518, patch_size=14)
        intrinsic = torch.tensor(
            [
                [1169.621094, 0.0, 646.295044],
                [0.0, 1167.105103, 489.927032],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        transformed = transform_intrinsics(intrinsic, transform)

        self.assertEqual((transform.output_width, transform.output_height), (518, 392))
        self.assertEqual((transform.crop_left, transform.crop_top), (0, 0))
        self.assertAlmostEqual(transform.scale_x, 518 / 1296)
        self.assertAlmostEqual(transform.scale_y, 392 / 968)
        self.assertAlmostEqual(float(transformed[0, 0]), 1169.621094 * 518 / 1296, places=4)
        self.assertAlmostEqual(float(transformed[1, 1]), 1167.105103 * 392 / 968, places=4)
        self.assertAlmostEqual(float(transformed[0, 2]), 646.295044 * 518 / 1296, places=4)
        self.assertAlmostEqual(float(transformed[1, 2]), 489.927032 * 392 / 968, places=4)

    def test_crop_transform_subtracts_center_crop_from_principal_point(self) -> None:
        transform = compute_image_preprocess_transform(1000, 2000, mode="crop", target_size=500, patch_size=10)
        intrinsic = torch.tensor([[100.0, 0.0, 500.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]])

        transformed = transform_intrinsics(intrinsic, transform)

        self.assertEqual(transform.resized_height, 1000)
        self.assertEqual(transform.output_height, 500)
        self.assertEqual(transform.crop_top, 250)
        self.assertAlmostEqual(float(transformed[1, 2]), 250.0)

    def test_batch_preprocess_transform_adds_center_padding_for_mixed_crop_heights(self) -> None:
        transforms = compute_batch_preprocess_transforms(
            [(1296, 968), (518, 518)],
            mode="crop",
            target_size=518,
            patch_size=14,
        )

        self.assertEqual((transforms[0].output_width, transforms[0].output_height), (518, 518))
        self.assertEqual(transforms[0].batch_pad_top, 63)
        self.assertEqual(transforms[1].batch_pad_top, 0)

    def test_depth_to_local_camera_points_matches_official_backprojection_math(self) -> None:
        depth = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        intrinsic = torch.eye(3)

        local, valid = depth_to_local_camera_points(depth, intrinsic)

        expected = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
                [[0.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ]
        )
        self.assertTrue(torch.allclose(local, expected))
        self.assertEqual(valid.tolist(), [[True, True], [True, True]])

    def test_depth_to_known_world_points_applies_camera_to_world_pose(self) -> None:
        depth = torch.ones((1, 1), dtype=torch.float32)
        intrinsic = torch.eye(3)
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, 3] = torch.tensor([10.0, 20.0, 30.0])

        world, valid = depth_to_known_world_points(depth, intrinsic, pose)

        self.assertTrue(valid[0, 0])
        self.assertTrue(torch.allclose(world[0, 0], torch.tensor([10.0, 20.0, 31.0])))

    def test_depth_views_to_known_world_points_preserves_per_view_ownership(self) -> None:
        depth = torch.ones((2, 1, 1), dtype=torch.float32)
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(2, 1, 1)
        poses = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
        poses[1, :3, 3] = torch.tensor([5.0, 0.0, 0.0])

        world, valid = depth_views_to_known_world_points(depth, intrinsics, poses, depth_scale=2.0)
        offsets = build_per_view_shape_offsets(["00000", "00010"], height=1, width=1)

        self.assertEqual(offsets[1]["point_offset"], 1)
        self.assertEqual(valid.tolist(), [[[True]], [[True]]])
        self.assertTrue(torch.allclose(world[:, 0, 0], torch.tensor([[0.0, 0.0, 2.0], [5.0, 0.0, 2.0]])))


if __name__ == "__main__":
    unittest.main()
