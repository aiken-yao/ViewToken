from __future__ import annotations

import math
import unittest

import torch

from viewtoken.oracle import (
    PinholeIntrinsics,
    build_visibility_masks,
    camera_pose_delta_to_observed,
    project_world_points,
    summarize_visibility_masks,
    visible_surface_mask,
)


def identity_pose() -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32)


def translated_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    pose = identity_pose()
    pose[:3, 3] = torch.tensor([x, y, z], dtype=torch.float32)
    return pose


def rotation_y_pose(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    pose = identity_pose()
    pose[:3, :3] = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=torch.float32,
    )
    return pose


class VisibilityTest(unittest.TestCase):
    def test_projection_boundaries_are_half_open(self) -> None:
        intrinsics = PinholeIntrinsics(fx=10.0, fy=10.0, cx=50.0, cy=50.0, width=100, height=100)
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [-5.0, 0.0, 1.0],
                [5.0, 0.0, 1.0],
                [0.0, -5.0, 1.0],
                [0.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )

        projection = project_world_points(points, identity_pose(), intrinsics)

        self.assertEqual(projection.in_frame_mask.tolist(), [True, True, False, True, False])
        self.assertEqual(projection.pixel_xy[0].tolist(), [50, 50])
        self.assertEqual(projection.pixel_xy[1].tolist(), [0, 50])
        self.assertEqual(projection.pixel_xy[3].tolist(), [50, 0])

    def test_camera_front_back_mask_uses_positive_z(self) -> None:
        intrinsics = PinholeIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=10, height=10)
        points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

        mask = visible_surface_mask(points, identity_pose(), intrinsics, depth_tolerance=0.0)

        self.assertEqual(mask.tolist(), [True, False])

    def test_depth_occlusion_keeps_nearest_same_pixel_surface(self) -> None:
        intrinsics = PinholeIntrinsics(fx=20.0, fy=20.0, cx=10.0, cy=10.0, width=20, height=20)
        points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        strict = visible_surface_mask(points, identity_pose(), intrinsics, depth_tolerance=0.0)
        tolerant = visible_surface_mask(points, identity_pose(), intrinsics, depth_tolerance=1.0)

        self.assertEqual(strict.tolist(), [False, True])
        self.assertEqual(tolerant.tolist(), [True, True])

    def test_repeated_camera_produces_full_overlap_and_no_novel_surface(self) -> None:
        intrinsics = PinholeIntrinsics(fx=10.0, fy=10.0, cx=50.0, cy=50.0, width=100, height=100)
        points = torch.tensor([[-1.0, 0.0, 5.0], [0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=torch.float32)

        masks = build_visibility_masks(points, [identity_pose()], identity_pose(), intrinsics, depth_tolerance=0.0)
        stats = summarize_visibility_masks(masks)

        self.assertEqual(stats.observed_count, 3)
        self.assertEqual(stats.candidate_count, 3)
        self.assertEqual(stats.overlap_count, 3)
        self.assertEqual(stats.novel_count, 0)

    def test_partial_overlap_reports_only_candidate_unobserved_points_as_novel(self) -> None:
        intrinsics = PinholeIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=5, height=5)
        points = torch.tensor([[-2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=torch.float32)

        masks = build_visibility_masks(
            points,
            [translated_pose(x=1.0)],
            translated_pose(x=-1.0),
            intrinsics,
            depth_tolerance=0.0,
        )
        stats = summarize_visibility_masks(masks)

        self.assertEqual(masks.observed.tolist(), [False, True, True])
        self.assertEqual(masks.candidate.tolist(), [True, True, False])
        self.assertEqual(stats.overlap_count, 1)
        self.assertEqual(stats.novel_count, 1)
        self.assertAlmostEqual(stats.candidate_novel_fraction or 0.0, 0.5)

    def test_no_overlap_keeps_candidate_surface_novel(self) -> None:
        intrinsics = PinholeIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=3, height=3)
        points = torch.tensor([[-3.0, 0.0, 1.0], [3.0, 0.0, 1.0]], dtype=torch.float32)

        masks = build_visibility_masks(
            points,
            [translated_pose(x=-3.0)],
            translated_pose(x=3.0),
            intrinsics,
            depth_tolerance=0.0,
        )
        stats = summarize_visibility_masks(masks)

        self.assertEqual(masks.observed.tolist(), [True, False])
        self.assertEqual(masks.candidate.tolist(), [False, True])
        self.assertEqual(stats.overlap_count, 0)
        self.assertEqual(stats.novel_count, 1)

    def test_camera_delta_reports_distance_and_view_direction_change(self) -> None:
        observed = {"origin": identity_pose()}
        candidate = rotation_y_pose(90.0)
        candidate[:3, 3] = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)

        delta = camera_pose_delta_to_observed(candidate, observed)

        self.assertAlmostEqual(delta.min_distance_to_observed_meters, 5.0)
        self.assertAlmostEqual(delta.min_view_direction_change_degrees, 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
