from __future__ import annotations

import unittest

import torch

from viewtoken.oracle import (
    OracleGainRecord,
    PointCloudMetrics,
    build_memory_id,
    compute_metric_gains,
    compute_pointcloud_metrics,
    scene_split,
    spearman_rank_correlation,
    summarize_oracle_audit,
    voxel_downsample_points,
)


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

    def test_voxel_downsample_averages_points_inside_voxel(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        downsampled = voxel_downsample_points(points, voxel_size=0.05)

        self.assertEqual(tuple(downsampled.shape), (2, 3))
        self.assertTrue(torch.allclose(downsampled[0], torch.tensor([0.005, 0.0, 0.0])))
        self.assertTrue(torch.allclose(downsampled[1], torch.tensor([1.0, 1.0, 1.0])))

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
                metadata={"candidate_sanity_tags": ["repeat_observed"]},
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
        self.assertEqual(summary["sanity_checks"]["repeat_observed"]["candidate_count"], 1)
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
