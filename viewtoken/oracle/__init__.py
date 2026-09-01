"""Offline oracle-gain label generation utilities."""

from .audit import (
    flatten_metric_gains,
    spearman_rank_correlation,
    spearman_rank_correlations,
    summarize_metric_gains,
    summarize_oracle_audit,
    summarize_sanity_checks,
)
from .dataset import OracleGainRecord, append_jsonl, build_memory_id, scene_split, write_jsonl
from .io import load_point_cloud, load_pose_matrix, view_id_from_path
from .metrics import (
    PointCloudMetrics,
    align_sim3_icp,
    compute_metric_gains,
    compute_pointcloud_metrics,
    voxel_downsample_points,
)

__all__ = [
    "OracleGainRecord",
    "PointCloudMetrics",
    "align_sim3_icp",
    "append_jsonl",
    "build_memory_id",
    "compute_metric_gains",
    "compute_pointcloud_metrics",
    "flatten_metric_gains",
    "load_point_cloud",
    "load_pose_matrix",
    "scene_split",
    "spearman_rank_correlation",
    "spearman_rank_correlations",
    "summarize_metric_gains",
    "summarize_oracle_audit",
    "summarize_sanity_checks",
    "view_id_from_path",
    "voxel_downsample_points",
    "write_jsonl",
]
