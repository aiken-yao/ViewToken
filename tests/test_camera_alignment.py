from __future__ import annotations

import math
import unittest

import torch

from viewtoken.oracle import (
    DegenerateCameraAnchorsError,
    camera_anchor_condition_number,
    camera_centers_from_camera_to_world,
    compute_metric_gains,
    compute_pointcloud_metrics,
    camera_centers_from_world_to_camera,
    estimate_camera_anchor_alignment,
    predicted_camera_centers_from_pose_enc,
)
from viewtoken.oracle.camera_alignment import VGGT_ROOT

if str(VGGT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(VGGT_ROOT))
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


def rotation_z(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


class CameraAlignmentTest(unittest.TestCase):
    def test_world_to_camera_center_uses_negative_r_transpose_t(self) -> None:
        center = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        rotation = rotation_z(30.0)
        translation = -(rotation @ center)
        extrinsics = torch.cat([rotation, translation[:, None]], dim=1).unsqueeze(0)

        decoded = camera_centers_from_world_to_camera(extrinsics)

        self.assertTrue(torch.allclose(decoded[0, 0], center, atol=1e-6))

    def test_scannet_camera_to_world_center_is_translation(self) -> None:
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, 3] = torch.tensor([0.5, 1.5, -0.25])

        center = camera_centers_from_camera_to_world(pose)

        self.assertEqual(center.shape, (1, 3))
        self.assertTrue(torch.allclose(center[0], pose[:3, 3]))

    def test_official_pose_encoding_decodes_to_world_to_camera_center(self) -> None:
        center = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32)
        rotation = rotation_z(-20.0)
        translation = -(rotation @ center)
        extrinsics = torch.cat([rotation, translation[:, None]], dim=1).reshape(1, 1, 3, 4)
        intrinsics = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
        intrinsics[..., 0, 0] = 400.0
        intrinsics[..., 1, 1] = 410.0
        intrinsics[..., 0, 2] = 259.0
        intrinsics[..., 1, 2] = 196.0
        pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=(392, 518))

        centers = predicted_camera_centers_from_pose_enc(pose_enc, image_size_hw=(392, 518))

        self.assertEqual(tuple(centers.shape), (1, 3))
        self.assertTrue(torch.allclose(centers[0], center, atol=1e-5))

    def test_camera_anchor_alignment_uses_shared_ids_only(self) -> None:
        predicted = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([0.0, 1.0, 0.0]),
            "99999": torch.tensor([10.0, 10.0, 10.0]),
        }
        gt = {
            "00000": torch.tensor([1.0, 2.0, 3.0]),
            "00010": torch.tensor([3.0, 2.0, 3.0]),
            "00020": torch.tensor([1.0, 4.0, 3.0]),
        }

        alignment = estimate_camera_anchor_alignment(
            predicted,
            gt,
            shared_anchor_ids=["00000", "00010", "00020"],
        )

        self.assertEqual(alignment.shared_anchor_ids, ["00000", "00010", "00020"])
        self.assertLessEqual(alignment.camera_rmse, 1e-6)
        self.assertNotIn("99999", alignment.aligned_predicted_centers)

    def test_camera_anchor_condition_number_detects_collinearity(self) -> None:
        collinear = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        non_collinear = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )

        self.assertTrue(math.isinf(camera_anchor_condition_number(collinear)))
        self.assertTrue(math.isfinite(camera_anchor_condition_number(non_collinear)))

    def test_camera_anchor_alignment_reports_predicted_and_gt_condition_numbers(self) -> None:
        predicted = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([0.0, 1.0, 0.0]),
        }
        gt = {
            "00000": torch.tensor([1.0, 2.0, 3.0]),
            "00010": torch.tensor([3.0, 2.0, 3.0]),
            "00020": torch.tensor([1.0, 4.0, 3.0]),
        }

        alignment = estimate_camera_anchor_alignment(
            predicted,
            gt,
            shared_anchor_ids=["00000", "00010", "00020"],
        )

        payload = alignment.to_dict()
        self.assertTrue(math.isfinite(payload["predicted_condition_number"]))
        self.assertTrue(math.isfinite(payload["gt_condition_number"]))
        self.assertEqual(payload["condition_number"], payload["predicted_condition_number"])

    def test_camera_anchor_alignment_rejects_collinear_predicted_anchors(self) -> None:
        predicted = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([2.0, 0.0, 0.0]),
        }
        gt = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([0.0, 1.0, 0.0]),
        }

        with self.assertRaises(DegenerateCameraAnchorsError):
            estimate_camera_anchor_alignment(
                predicted,
                gt,
                shared_anchor_ids=["00000", "00010", "00020"],
            )

    def test_camera_anchor_alignment_rejects_non_finite_centers(self) -> None:
        predicted = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([float("nan"), 1.0, 0.0]),
        }
        gt = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([0.0, 1.0, 0.0]),
        }

        with self.assertRaises(DegenerateCameraAnchorsError):
            estimate_camera_anchor_alignment(
                predicted,
                gt,
                shared_anchor_ids=["00000", "00010", "00020"],
            )

    def test_camera_anchor_identical_cloud_gain_is_zero(self) -> None:
        predicted = {
            "00000": torch.tensor([0.0, 0.0, 0.0]),
            "00010": torch.tensor([1.0, 0.0, 0.0]),
            "00020": torch.tensor([0.0, 1.0, 0.0]),
        }
        gt = {
            "00000": torch.tensor([1.0, 2.0, 3.0]),
            "00010": torch.tensor([3.0, 2.0, 3.0]),
            "00020": torch.tensor([1.0, 4.0, 3.0]),
        }
        alignment = estimate_camera_anchor_alignment(
            predicted,
            gt,
            shared_anchor_ids=["00000", "00010", "00020"],
        )
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.0, 0.4, 0.2],
                [0.2, 0.3, 0.5],
            ],
            dtype=torch.float32,
        )
        aligned = alignment.transform.apply(points)
        baseline = compute_pointcloud_metrics(
            aligned, aligned, thresholds=(0.05, 0.1), max_points=None
        )
        candidate = compute_pointcloud_metrics(
            aligned, aligned, thresholds=(0.05, 0.1), max_points=None
        )

        gains = compute_metric_gains(baseline, candidate)

        self.assertAlmostEqual(gains["chamfer"], 0.0)
        self.assertAlmostEqual(gains["accuracy"], 0.0)
        self.assertAlmostEqual(gains["completeness"], 0.0)
        self.assertAlmostEqual(gains["coverage"], 0.0)
        self.assertAlmostEqual(gains["fscore"]["0.05"], 0.0)
        self.assertAlmostEqual(gains["fscore"]["0.1"], 0.0)


if __name__ == "__main__":
    unittest.main()
