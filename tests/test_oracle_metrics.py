from __future__ import annotations

import math
import unittest

import torch

from viewtoken.oracle import (
    OracleGainRecord,
    PointCloudMetrics,
    SimilarityTransform,
    align_sim3_icp_with_diagnostics,
    apply_similarity_transform,
    build_memory_id,
    compute_metric_gains,
    compute_pointcloud_metrics,
    compute_pointcloud_residual_diagnostics,
    estimate_similarity_transform,
    identical_cloud_gain_check,
    sample_point_indices,
    scene_split,
    spearman_rank_correlation,
    summarize_oracle_audit,
    voxel_downsample_points,
)


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


def rotation_xyz(x_degrees: float, y_degrees: float, z_degrees: float) -> torch.Tensor:
    x = math.radians(x_degrees)
    y = math.radians(y_degrees)
    z = math.radians(z_degrees)
    rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(x), -math.sin(x)],
            [0.0, math.sin(x), math.cos(x)],
        ],
        dtype=torch.float32,
    )
    ry = torch.tensor(
        [
            [math.cos(y), 0.0, math.sin(y)],
            [0.0, 1.0, 0.0],
            [-math.sin(y), 0.0, math.cos(y)],
        ],
        dtype=torch.float32,
    )
    rz = torch.tensor(
        [
            [math.cos(z), -math.sin(z), 0.0],
            [math.sin(z), math.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return rz @ ry @ rx


class OracleMetricsTest(unittest.TestCase):
    def test_identical_point_cloud_has_perfect_metrics(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )

        metrics = compute_pointcloud_metrics(
            points,
            points,
            thresholds=(0.01,),
            coverage_radius=0.01,
            max_points=None,
        )

        self.assertAlmostEqual(metrics.chamfer, 0.0)
        self.assertAlmostEqual(metrics.accuracy, 0.0)
        self.assertAlmostEqual(metrics.completeness, 0.0)
        self.assertAlmostEqual(metrics.fscore[0.01], 1.0)
        self.assertAlmostEqual(metrics.coverage, 1.0)

    def test_metric_gains_use_larger_is_better_convention(self) -> None:
        baseline = PointCloudMetrics(
            chamfer=2.0,
            accuracy=1.5,
            completeness=2.5,
            fscore={0.1: 0.25},
            coverage=0.3,
        )
        candidate = PointCloudMetrics(
            chamfer=1.5,
            accuracy=1.0,
            completeness=2.25,
            fscore={0.1: 0.5},
            coverage=0.4,
        )

        gains = compute_metric_gains(baseline, candidate)

        self.assertAlmostEqual(gains["chamfer"], 0.5)
        self.assertAlmostEqual(gains["accuracy"], 0.5)
        self.assertAlmostEqual(gains["completeness"], 0.25)
        self.assertAlmostEqual(gains["fscore"]["0.1"], 0.25)
        self.assertAlmostEqual(gains["coverage"], 0.1)


    def test_hash_point_sampling_is_stable_and_configurable(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        first = sample_point_indices(points, max_points=3, seed=7, method="hash")
        second = sample_point_indices(points, max_points=3, seed=7, method="hash")
        all_indices = sample_point_indices(points, max_points=0, seed=7, method="hash")

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), 3)
        self.assertTrue(torch.equal(all_indices, torch.arange(points.shape[0])))

    def test_voxel_downsample_averages_points_inside_voxel(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        downsampled = voxel_downsample_points(points, voxel_size=0.05)

        self.assertEqual(tuple(downsampled.shape), (2, 3))
        self.assertTrue(torch.allclose(downsampled[0], torch.tensor([0.005, 0.0, 0.0])))
        self.assertTrue(torch.allclose(downsampled[1], torch.tensor([1.0, 1.0, 1.0])))

    def test_identical_cloud_alignment_pipeline_has_zero_gain(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.0, 0.4, 0.2],
                [0.2, 0.3, 0.5],
                [0.7, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )

        check = identical_cloud_gain_check(
            points=points,
            target=points,
            alignment="sim3_icp",
            seed=0,
            thresholds=(0.05, 0.1),
            coverage_radius=0.05,
            max_points=None,
            voxel_size=None,
        )

        self.assertLessEqual(check["max_abs_gain"], 1e-7)

    def test_known_sim3_recovery_full_overlap(self) -> None:
        generator = torch.Generator().manual_seed(7)
        source = torch.rand((64, 3), generator=generator)
        expected = SimilarityTransform(
            scale=1.7,
            rotation=rotation_z(23.0),
            translation=torch.tensor([0.4, -0.2, 0.8], dtype=torch.float32),
        )
        target = expected.apply(source)

        recovered = estimate_similarity_transform(source, target)

        self.assertAlmostEqual(recovered.scale, expected.scale, places=5)
        self.assertTrue(torch.allclose(recovered.rotation, expected.rotation, atol=1e-5))
        self.assertTrue(torch.allclose(recovered.translation, expected.translation, atol=1e-5))
        self.assertTrue(torch.allclose(recovered.apply(source), target, atol=1e-5))

    def test_known_sim3_recovery_partial_overlap_correspondences(self) -> None:
        generator = torch.Generator().manual_seed(9)
        source = torch.rand((96, 3), generator=generator)
        expected = SimilarityTransform(
            scale=0.6,
            rotation=rotation_z(-17.0),
            translation=torch.tensor([-0.3, 0.5, 0.2], dtype=torch.float32),
        )
        target = expected.apply(source)
        partial = torch.arange(0, source.shape[0], 3)

        recovered = estimate_similarity_transform(source[partial], target[partial])

        self.assertAlmostEqual(recovered.scale, expected.scale, places=5)
        self.assertTrue(torch.allclose(recovered.rotation, expected.rotation, atol=1e-5))
        self.assertTrue(torch.allclose(recovered.translation, expected.translation, atol=1e-5))

    def test_known_sim3_recovery_arbitrary_3d_rotation(self) -> None:
        generator = torch.Generator().manual_seed(11)
        source = torch.rand((80, 3), generator=generator)
        expected = SimilarityTransform(
            scale=2.25,
            rotation=rotation_xyz(31.0, -14.0, 67.0),
            translation=torch.tensor([0.8, -0.4, 1.2], dtype=torch.float32),
        )
        target = expected.apply(source)

        recovered = estimate_similarity_transform(source, target)

        self.assertAlmostEqual(recovered.scale, expected.scale, places=5)
        self.assertTrue(torch.allclose(recovered.rotation, expected.rotation, atol=1e-5))
        self.assertTrue(torch.allclose(recovered.translation, expected.translation, atol=1e-5))

    def test_known_sim3_recovery_non_coplanar_four_points(self) -> None:
        source = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        expected = SimilarityTransform(
            scale=0.75,
            rotation=rotation_xyz(-20.0, 35.0, 12.0),
            translation=torch.tensor([-0.5, 0.7, 0.25], dtype=torch.float32),
        )
        target = expected.apply(source)

        recovered = estimate_similarity_transform(source, target)

        self.assertAlmostEqual(recovered.scale, expected.scale, places=5)
        self.assertTrue(torch.allclose(recovered.rotation, expected.rotation, atol=1e-5))
        self.assertTrue(torch.allclose(recovered.apply(source), target, atol=1e-5))

    def test_sim3_reflection_rejection_uses_signed_singular_values_for_scale(self) -> None:
        source = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            dtype=torch.float32,
        )
        target = source.clone()
        target[:, 0] *= -1.0

        recovered = estimate_similarity_transform(source, target)

        self.assertGreater(torch.linalg.det(recovered.rotation).item(), 0.999)
        self.assertLess(recovered.scale, 0.5)
        self.assertGreater(
            torch.linalg.norm(recovered.apply(source) - target, dim=1).mean().item(),
            1.0,
        )

    def test_pointcloud_residual_diagnostics_reports_all_thresholds(self) -> None:
        source = torch.tensor(
            [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=torch.float32,
        )
        target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        diagnostics = compute_pointcloud_residual_diagnostics(
            source,
            target,
            inlier_thresholds=(0.02, 0.05, 0.1),
            max_points=None,
        ).to_dict()

        self.assertEqual(diagnostics["source_sample_count"], 3)
        self.assertAlmostEqual(diagnostics["inlier_ratios"]["0.02"], 1.0 / 3.0)
        self.assertAlmostEqual(diagnostics["inlier_ratios"]["0.05"], 2.0 / 3.0)
        self.assertAlmostEqual(diagnostics["inlier_ratios"]["0.1"], 2.0 / 3.0)

    def test_alignment_diagnostics_report_transform_and_residuals(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.0, 0.4, 0.2],
                [0.2, 0.3, 0.5],
                [0.7, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )

        result = align_sim3_icp_with_diagnostics(
            points,
            points,
            seed=0,
            sample_size=None,
            inlier_threshold=1e-4,
        )

        diagnostics = result.diagnostics.to_dict()
        self.assertIn("transform", diagnostics)
        self.assertLessEqual(diagnostics["residual_rmse"], 1e-6)
        self.assertEqual(diagnostics["inlier_ratio"], 1.0)

    def test_spearman_rank_correlation_handles_order_and_ties(self) -> None:
        self.assertAlmostEqual(
            spearman_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            -1.0,
        )
        self.assertIsNone(spearman_rank_correlation([1.0, 1.0], [2.0, 3.0]))

    def test_oracle_audit_summary_reports_distribution_and_sanity(self) -> None:
        baseline = PointCloudMetrics(
            chamfer=1.0,
            accuracy=0.6,
            completeness=0.4,
            fscore={0.05: 0.3, 0.1: 0.5},
            coverage=0.2,
        )
        records = [
            OracleGainRecord(
                scene_id="scene",
                split="train",
                memory_id="memory",
                observed_view_ids=["00000"],
                candidate_view_id="00000",
                candidate_pose=[],
                baseline_metrics=baseline,
                candidate_metrics=baseline,
                reconstruction_paths={},
                metadata={"candidate_sanity_tags": ["duplicate_input_sensitivity"]},
            ),
            OracleGainRecord(
                scene_id="scene",
                split="train",
                memory_id="memory",
                observed_view_ids=["00000"],
                candidate_view_id="00010",
                candidate_pose=[],
                baseline_metrics=baseline,
                candidate_metrics=PointCloudMetrics(
                    chamfer=0.8,
                    accuracy=0.5,
                    completeness=0.3,
                    fscore={0.05: 0.4, 0.1: 0.7},
                    coverage=0.35,
                ),
                reconstruction_paths={},
                metadata={"candidate_sanity_tags": ["new_area"]},
            ),
        ]

        summary = summarize_oracle_audit(records, observed_view_ids=["00000"])

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["held_out_candidate_count"], 1)
        self.assertEqual(summary["held_out_metrics"]["coverage"]["oracle_best_candidate"], "00010")
        self.assertAlmostEqual(
            summary["held_out_metrics"]["coverage"]["random_candidate_mean_gain"],
            0.15,
        )
        self.assertEqual(summary["sanity_checks"]["duplicate_input_sensitivity"]["candidate_count"], 1)
        self.assertIn("chamfer|coverage", summary["spearman_rank_correlation"])

    def test_memory_id_and_split_are_scene_level_deterministic(self) -> None:
        self.assertEqual(
            build_memory_id("scene", ["00000", "00010"]),
            build_memory_id("scene", ["00000", "00010"]),
        )
        self.assertEqual(scene_split("scene_a"), scene_split("scene_a"))
        self.assertIn(scene_split("scene_a"), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
