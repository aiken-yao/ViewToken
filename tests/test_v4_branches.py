from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from viewtoken.oracle import (
    V4CacheData,
    compare_branch_reconstructions,
    heldout_candidate_pose_diagnostics,
    recover_v4_branch_points,
)


def make_cache(candidate_center=(2.0, 2.0, 0.0), calibrated_fx=2.0) -> V4CacheData:
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], candidate_center],
        dtype=torch.float32,
    )
    extrinsics = torch.zeros((4, 3, 4), dtype=torch.float32)
    extrinsics[:, :3, :3] = torch.eye(3)
    extrinsics[:, :3, 3] = -centers
    depth = torch.ones((4, 1, 2), dtype=torch.float32)
    predicted = torch.eye(3).repeat(4, 1, 1)
    calibrated = predicted.clone()
    calibrated[:, 0, 0] = calibrated_fx
    per_view_world = torch.zeros((4, 1, 2, 3), dtype=torch.float32)
    per_view_world[..., 2] = 1.0
    return V4CacheData(
        reconstruction_dir=Path(tempfile.gettempdir()),
        metadata={"input_shape": [4, 3, 1, 2], "per_view_shape_offsets": []},
        image_view_ids=["00000", "00010", "00020", "99999"],
        points=per_view_world.reshape(-1, 3),
        confidence=torch.ones(8),
        per_view_world_points=per_view_world,
        per_view_confidence=torch.ones((4, 1, 2)),
        world_to_camera_extrinsics=extrinsics,
        predicted_intrinsics=predicted,
        depth=depth,
        depth_conf=torch.ones_like(depth),
        transformed_gt_intrinsics=calibrated,
    )


def gt_poses() -> torch.Tensor:
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, :3, 3] = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [1.0, 4.0, 3.0], [5.0, 6.0, 3.0]]
    )
    return poses


class V4BranchesTest(unittest.TestCase):
    def test_candidate_pose_does_not_change_observed_only_scale(self) -> None:
        first = make_cache(candidate_center=(2.0, 2.0, 0.0))
        second = make_cache(candidate_center=(100.0, -50.0, 7.0))

        _, first_meta = recover_v4_branch_points(first, gt_poses(), ["00000", "00010", "00020"], "C")
        _, second_meta = recover_v4_branch_points(second, gt_poses(), ["00000", "00010", "00020"], "C")

        self.assertAlmostEqual(first_meta["depth_scale"], 2.0, places=5)
        self.assertAlmostEqual(first_meta["depth_scale"], second_meta["depth_scale"], places=6)

    def test_predicted_and_calibrated_intrinsics_produce_distinct_branches(self) -> None:
        cache = make_cache(calibrated_fx=2.0)

        branch_c, _ = recover_v4_branch_points(cache, gt_poses(), ["00000", "00010", "00020"], "C")
        branch_d, _ = recover_v4_branch_points(cache, gt_poses(), ["00000", "00010", "00020"], "D")

        self.assertFalse(torch.allclose(branch_c, branch_d))

    def test_identical_reconstructions_have_zero_region_gain(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        result = compare_branch_reconstructions(
            points,
            points,
            points,
            observed_mask=torch.tensor([True, True, False, False]),
            novel_mask=torch.tensor([False, False, True, True]),
            max_prediction_points=None,
        )

        for region in ("observed", "novel"):
            for threshold in ("0.05", "0.1", "0.2"):
                self.assertEqual(result[region]["covered"][threshold]["covered_count_gain"], 0)
        self.assertAlmostEqual(result["accuracy"]["outlier_ratio_gain"], 0.0)

    def test_heldout_diagnostics_use_only_observed_alignment_ids(self) -> None:
        result = heldout_candidate_pose_diagnostics(
            make_cache(), gt_poses(), ["00000", "00010", "00020"], "99999"
        )

        self.assertEqual(result["alignment_uses_observed_ids_only"], ["00000", "00010", "00020"])
        self.assertGreaterEqual(result["center_error_meters"], 0.0)


if __name__ == "__main__":
    unittest.main()
