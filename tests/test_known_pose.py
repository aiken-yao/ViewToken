from __future__ import annotations

import unittest

import torch

from viewtoken.oracle import (
    KnownPoseCacheEligibilityError,
    fuse_cached_points_with_known_poses,
    inspect_per_view_flatten_metadata,
    local_camera_points_to_known_world,
    predicted_world_to_local_camera_points,
    reshape_flattened_points_by_view,
)


def eligible_metadata() -> dict[str, object]:
    return {
        "input_shape": [2, 3, 2, 3],
        "raw_world_point_count_before_filter": 12,
        "filtered_world_point_count": 12,
        "point_count": 12,
        "max_reconstruction_points": None,
        "reconstruction_sample_method": "none",
        "min_world_point_confidence": 0.0,
        "finite_world_point_ratio_before_filter": 1.0,
        "valid_world_point_ratio": 1.0,
    }


class KnownPoseTest(unittest.TestCase):
    def test_flatten_metadata_accepts_complete_unfiltered_cache(self) -> None:
        flatten = inspect_per_view_flatten_metadata(eligible_metadata())

        self.assertEqual(flatten.view_count, 2)
        self.assertEqual(flatten.height, 2)
        self.assertEqual(flatten.width, 3)
        self.assertEqual(flatten.expected_point_count, 12)

    def test_flatten_metadata_rejects_filtered_or_sampled_cache(self) -> None:
        cases = [
            {"filtered_world_point_count": 11},
            {"point_count": 11},
            {"max_reconstruction_points": 100},
            {"reconstruction_sample_method": "hash"},
            {"min_world_point_confidence": 0.1},
            {"finite_world_point_ratio_before_filter": 0.99},
            {"valid_world_point_ratio": 0.99},
        ]
        for update in cases:
            metadata = eligible_metadata()
            metadata.update(update)
            with self.subTest(update=update):
                with self.assertRaises(KnownPoseCacheEligibilityError):
                    inspect_per_view_flatten_metadata(metadata)

    def test_reshape_flattened_points_by_view_preserves_row_major_view_order(self) -> None:
        points = torch.arange(36, dtype=torch.float32).reshape(12, 3)

        reshaped, flatten = reshape_flattened_points_by_view(points, eligible_metadata())

        self.assertEqual(tuple(reshaped.shape), (2, 2, 3, 3))
        self.assertEqual(flatten.expected_point_count, 12)
        self.assertTrue(torch.equal(reshaped[0, 0, 0], points[0]))
        self.assertTrue(torch.equal(reshaped[1, 0, 0], points[6]))

    def test_predicted_world_to_local_camera_points_applies_world_to_camera_extrinsic(self) -> None:
        points = torch.tensor([[[[2.0, 3.0, 4.0]]]], dtype=torch.float32)
        extrinsics = torch.tensor(
            [[[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0]]],
            dtype=torch.float32,
        )

        local = predicted_world_to_local_camera_points(points, extrinsics)

        self.assertTrue(torch.allclose(local[0, 0, 0], torch.tensor([12.0, 23.0, 34.0])))

    def test_local_camera_points_to_known_world_applies_scale_and_pose(self) -> None:
        local = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
        poses = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
        poses[0, :3, 3] = torch.tensor([10.0, 20.0, 30.0])

        world = local_camera_points_to_known_world(local, poses, depth_scale=2.0)

        self.assertTrue(torch.allclose(world[0, 0, 0], torch.tensor([12.0, 24.0, 36.0])))

    def test_fuse_cached_points_with_known_poses_returns_flat_metric_points(self) -> None:
        metadata = {
            "input_shape": [1, 3, 1, 2],
            "raw_world_point_count_before_filter": 2,
            "filtered_world_point_count": 2,
            "point_count": 2,
            "max_reconstruction_points": None,
            "reconstruction_sample_method": "none",
            "min_world_point_confidence": 0.0,
            "finite_world_point_ratio_before_filter": 1.0,
            "valid_world_point_ratio": 1.0,
        }
        points = torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 4.0]], dtype=torch.float32)
        extrinsics = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )
        poses = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)

        fused, flatten = fuse_cached_points_with_known_poses(
            points,
            metadata,
            world_to_camera_extrinsics=extrinsics,
            camera_to_world_poses=poses,
            depth_scale=2.0,
        )

        self.assertEqual(flatten["expected_point_count"], 2)
        self.assertTrue(
            torch.allclose(
                fused,
                torch.tensor([[3.0, 0.0, 6.0], [5.0, 0.0, 10.0]], dtype=torch.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()
