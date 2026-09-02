"""Stage-B camera-anchor alignment audit using cached reconstructions only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt"
for import_root in (PROJECT_ROOT, VGGT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from viewtoken.oracle import (  # noqa: E402
    DegenerateCameraAnchorsError,
    OracleGainRecord,
    PointCloudMetrics,
    ReconstructionCacheValidationError,
    align_sim3_icp_with_diagnostics,
    compute_pointcloud_metrics,
    compute_pointcloud_residual_diagnostics,
    estimate_camera_anchor_alignment,
    load_gt_camera_centers_by_view_id,
    load_point_cloud,
    load_reconstruction_camera_centers,
    summarize_oracle_audit,
    validate_reconstruction_cache,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audit-records", type=Path, default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--gt-pose-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    parser.add_argument("--fscore-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--coverage-radius", type=float, default=None)
    parser.add_argument("--voxel-downsample-size", type=float, default=None)
    parser.add_argument("--max-metric-points", type=int, default=None)
    parser.add_argument("--metric-sample-method", choices=("random", "hash", "none"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-camera-anchor-condition-number", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return config


def value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    cli_value = getattr(args, name.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.lower() in {"", "none", "null", "all", "unlimited"}:
        return None
    parsed = int(raw)
    return None if parsed <= 0 else parsed


def resolve_path(raw: Any, label: str, must_exist: bool = True) -> Path:
    if raw is None:
        raise ValueError(f"{label} is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics_from_dict(payload: dict[str, Any]) -> PointCloudMetrics:
    return PointCloudMetrics(
        chamfer=float(payload["chamfer"]),
        accuracy=float(payload["accuracy"]),
        completeness=float(payload["completeness"]),
        fscore={float(key): float(value) for key, value in payload["fscore"].items()},
        coverage=float(payload["coverage"]),
    )


def infer_gt_pose_dir(records: list[dict[str, Any]]) -> Path:
    if not records:
        raise ValueError("Cannot infer GT pose dir from empty records")
    metadata = records[0].get("metadata", {})
    reconstruction = metadata.get("baseline_reconstruction", {})
    image_paths = reconstruction.get("image_paths", [])
    if not image_paths:
        raise ValueError("gt-pose-dir is required when records do not contain image_paths")
    return Path(image_paths[0]).expanduser().resolve().parent


def expected_reconstruction_fingerprint(record: dict[str, Any], role: str) -> str | None:
    metadata = record.get("metadata", {})
    reconstruction = metadata.get(f"{role}_reconstruction", {})
    fingerprint = reconstruction.get("cache_fingerprint")
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def collect_invalid_reconstruction_caches(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record_index, record in enumerate(records):
        for role, path_value in sorted(record.get("reconstruction_paths", {}).items()):
            reconstruction_dir = Path(path_value).expanduser().resolve().parent
            key = str(reconstruction_dir)
            if key in seen:
                continue
            seen.add(key)
            expected_fingerprint = expected_reconstruction_fingerprint(record, role)
            try:
                validate_reconstruction_cache(
                    reconstruction_dir, expected_fingerprint=expected_fingerprint
                )
            except ReconstructionCacheValidationError as exc:
                invalid.append(
                    {
                        "record_index": record_index,
                        "role": role,
                        "reconstruction_dir": str(reconstruction_dir),
                        "expected_fingerprint_present": expected_fingerprint is not None,
                        "errors": exc.errors,
                    }
                )
    return invalid


def count_missing_pose_enc(invalid_caches: list[dict[str, Any]]) -> int:
    return sum(
        "missing required artifact: pose_enc.pt" in item.get("errors", [])
        for item in invalid_caches
    )


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    status = report["status"]
    if status == "blocked_invalid_reconstruction_cache":
        invalid = report["invalid_reconstruction_caches"]
        preview_lines = []
        for item in invalid[:12]:
            preview_lines.append(
                f"- `{item['reconstruction_dir']}`: " + "; ".join(item["errors"])
            )
        if len(invalid) > 12:
            preview_lines.append(f"- ... {len(invalid) - 12} more")
        invalid_preview = "\n".join(preview_lines)
        text = f"""# Stage B Camera-Anchor Alignment Audit

Status: `blocked_invalid_reconstruction_cache`

This Stage-B audit did not run VGGT and did not create new reconstructions. It attempted to reuse existing audit cached reconstructions, but the caches are not complete `oracle-reconstruction-v3` artifacts.

## Inputs

- Audit records: `{report['inputs']['audit_records']}`
- Target points: `{report['inputs']['target_points']}`
- GT pose dir: `{report['inputs']['gt_pose_dir']}`
- Attempted records: `{report['attempted_record_count']}`
- Invalid reconstruction caches: `{len(invalid)}`
- Missing `pose_enc.pt` artifacts: `{report['missing_pose_enc_count']}`

## Invalid Caches

{invalid_preview}

## Interpretation

Stage B requires each baseline and observed-plus-candidate reconstruction cache to store `points.pt`, `confidence.pt`, `pose_enc.pt`, `metadata.json`, schema `oracle-reconstruction-v3`, and a valid cache fingerprint. The existing audit20 cache predates those requirements, so the correct behavior is to stop here rather than rerun VGGT silently or mix pose encodings from a separate run.
"""
    elif status == "blocked_degenerate_camera_anchors":
        text = f"""# Stage B Camera-Anchor Alignment Audit

Status: `blocked_degenerate_camera_anchors`

This Stage-B audit did not run VGGT and did not create new reconstructions. Cache integrity passed, but the shared observed camera anchors could not define a reliable camera-anchored Sim(3).

## Inputs

- Audit records: `{report['inputs']['audit_records']}`
- Target points: `{report['inputs']['target_points']}`
- GT pose dir: `{report['inputs']['gt_pose_dir']}`
- Attempted records: `{report['attempted_record_count']}`
- Max camera-anchor condition number: `{report['config']['max_camera_anchor_condition_number']}`

## Error

`{report['error']}`

## Interpretation

Stage B must fail closed when predicted or GT camera centers are non-finite, coincident, collinear, or too ill-conditioned. Candidate cameras are still not allowed to become alignment anchors; the next valid fix is to use better observed anchors, such as four initial observed views, or add an orientation-constrained alignment protocol.
"""
    else:
        held = report["camera_anchor_summary"]["held_out_metrics"]
        rows = []
        for metric_name, stats in held.items():
            rows.append(
                f"| `{metric_name}` | `{stats['mean']:.8f}` | `{stats['positive_gain_ratio']:.3f}` | "
                f"`{stats['oracle_best_candidate']}` | `{stats['oracle_best_gain']:.8f}` |"
            )
        diagnostic_rows = []
        for name, stats in report.get("point_cloud_diagnostic_summary", {}).items():
            diagnostic_rows.append(
                f"| `{name}` | `{stats['residual_mean']:.6f}` | `{stats['residual_median']:.6f}` | "
                f"`{stats['residual_rmse']:.6f}` | `{stats['residual_max']:.6f}` | "
                f"`{stats['inlier_ratios'].get('0.05', 0.0):.6f}` |"
            )
        text = f"""# Stage B Camera-Anchor Alignment Audit

Status: `complete`

This audit reuses cached reconstructions and aligns each reconstruction to ScanNet GT using only the shared observed camera centers. Candidate cameras are not used as alignment anchors.

| metric | mean gain | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---|---:|
{chr(10).join(rows)}

## Point-Cloud Diagnostics

| protocol | residual mean | residual median | residual RMSE | residual max | inlier@0.05m |
|---|---:|---:|---:|---:|---:|
{chr(10).join(diagnostic_rows)}

Output records: `{report['outputs']['camera_anchor_records']}`
Summary JSON: `{report['outputs']['camera_anchor_summary']}`
"""
    path.write_text(text)



def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_point_residual_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_keys = sorted(
        {
            key
            for payload in payloads
            for key in payload.get("inlier_ratios", {})
        },
        key=float,
    )
    return {
        "record_count": len(payloads),
        "residual_mean": _mean([float(payload["residual_mean"]) for payload in payloads]),
        "residual_median": _mean([float(payload["residual_median"]) for payload in payloads]),
        "residual_rmse": _mean([float(payload["residual_rmse"]) for payload in payloads]),
        "residual_max": _mean([float(payload["residual_max"]) for payload in payloads]),
        "inlier_ratios": {
            key: _mean([
                float(payload.get("inlier_ratios", {}).get(key, 0.0))
                for payload in payloads
            ])
            for key in ratio_keys
        },
    }


def summarize_point_cloud_diagnostics(records: list[OracleGainRecord]) -> dict[str, Any]:
    field_names = (
        "baseline_camera_anchor_point_residuals",
        "candidate_camera_anchor_point_residuals",
        "baseline_old_free_icp_point_residuals",
        "candidate_old_free_icp_point_residuals",
    )
    summary = {}
    for field_name in field_names:
        payloads = [
            record.metadata[field_name]
            for record in records
            if field_name in record.metadata
        ]
        if payloads:
            summary[field_name] = summarize_point_residual_payloads(payloads)
    return summary


def compute_camera_anchor_records(
    records: list[dict[str, Any]],
    target_points_path: Path,
    gt_pose_dir: Path,
    point_stride: int | None,
    thresholds: tuple[float, ...],
    coverage_radius: float,
    voxel_size: float | None,
    max_metric_points: int | None,
    seed: int,
    metric_sample_method: str,
    max_camera_anchor_condition_number: float,
) -> list[OracleGainRecord]:
    target_points = load_point_cloud(target_points_path, point_stride=point_stride)
    output_records: list[OracleGainRecord] = []
    for record in records:
        observed_ids = list(record["observed_view_ids"])
        gt_centers = load_gt_camera_centers_by_view_id(observed_ids, gt_pose_dir)

        baseline_points_path = Path(record["reconstruction_paths"]["baseline"]).expanduser().resolve()
        candidate_points_path = Path(record["reconstruction_paths"]["candidate"]).expanduser().resolve()
        baseline_dir = baseline_points_path.parent
        candidate_dir = candidate_points_path.parent
        baseline_centers = load_reconstruction_camera_centers(baseline_dir)
        candidate_centers = load_reconstruction_camera_centers(candidate_dir)

        baseline_alignment = estimate_camera_anchor_alignment(
            baseline_centers,
            gt_centers,
            shared_anchor_ids=observed_ids,
            max_condition_number=max_camera_anchor_condition_number,
        )
        candidate_alignment = estimate_camera_anchor_alignment(
            candidate_centers,
            gt_centers,
            shared_anchor_ids=observed_ids,
            max_condition_number=max_camera_anchor_condition_number,
        )

        baseline_points = load_point_cloud(baseline_points_path)
        candidate_points = load_point_cloud(candidate_points_path)
        baseline_aligned_points = baseline_alignment.transform.apply(baseline_points)
        candidate_aligned_points = candidate_alignment.transform.apply(candidate_points)
        baseline_point_residuals = compute_pointcloud_residual_diagnostics(
            baseline_aligned_points,
            target_points,
            inlier_thresholds=thresholds,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )
        candidate_point_residuals = compute_pointcloud_residual_diagnostics(
            candidate_aligned_points,
            target_points,
            inlier_thresholds=thresholds,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )
        baseline_free_icp = align_sim3_icp_with_diagnostics(
            baseline_points,
            target_points,
            seed=seed,
            sample_size=4096,
            inlier_threshold=0.05,
        )
        candidate_free_icp = align_sim3_icp_with_diagnostics(
            candidate_points,
            target_points,
            seed=seed,
            sample_size=4096,
            inlier_threshold=0.05,
        )
        baseline_free_icp_point_residuals = compute_pointcloud_residual_diagnostics(
            baseline_free_icp.points,
            target_points,
            inlier_thresholds=thresholds,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )
        candidate_free_icp_point_residuals = compute_pointcloud_residual_diagnostics(
            candidate_free_icp.points,
            target_points,
            inlier_thresholds=thresholds,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )
        baseline_metrics = compute_pointcloud_metrics(
            baseline_aligned_points,
            target_points,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )
        candidate_metrics = compute_pointcloud_metrics(
            candidate_aligned_points,
            target_points,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
            sample_method=metric_sample_method,
        )

        metadata = dict(record.get("metadata", {}))
        metadata["alignment_protocol"] = "camera_anchor_sim3_shared_observed"
        metadata["candidate_used_as_alignment_anchor"] = False
        metadata["shared_anchor_ids"] = observed_ids
        metadata["baseline_camera_anchor_alignment"] = baseline_alignment.to_dict()
        metadata["candidate_camera_anchor_alignment"] = candidate_alignment.to_dict()
        metadata["baseline_camera_anchor_point_residuals"] = baseline_point_residuals.to_dict()
        metadata["candidate_camera_anchor_point_residuals"] = candidate_point_residuals.to_dict()
        metadata["baseline_old_free_icp_alignment_diagnostics"] = baseline_free_icp.diagnostics.to_dict()
        metadata["candidate_old_free_icp_alignment_diagnostics"] = candidate_free_icp.diagnostics.to_dict()
        metadata["baseline_old_free_icp_point_residuals"] = baseline_free_icp_point_residuals.to_dict()
        metadata["candidate_old_free_icp_point_residuals"] = candidate_free_icp_point_residuals.to_dict()
        metadata["old_free_icp_metrics"] = {
            "baseline": record.get("baseline_metrics"),
            "candidate": record.get("candidate_metrics"),
            "gains": record.get("gains"),
        }
        output_records.append(
            OracleGainRecord(
                scene_id=record["scene_id"],
                split=record["split"],
                memory_id=record["memory_id"],
                observed_view_ids=observed_ids,
                candidate_view_id=record["candidate_view_id"],
                candidate_pose=record["candidate_pose"],
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                reconstruction_paths=record["reconstruction_paths"],
                metadata=metadata,
            )
        )
    return output_records


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    audit_records_path = resolve_path(value(args, config, "audit-records", None), "audit-records")
    target_points_path = resolve_path(value(args, config, "target-points", None), "target-points")
    output_dir = resolve_path(value(args, config, "output-dir", "outputs/oracle_calibration/stage_b_camera_anchor"), "output-dir", must_exist=False)
    records = load_jsonl(audit_records_path)
    gt_pose_dir_raw = value(args, config, "gt-pose-dir", None)
    gt_pose_dir = infer_gt_pose_dir(records) if gt_pose_dir_raw is None else resolve_path(gt_pose_dir_raw, "gt-pose-dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    common_report = {
        "inputs": {
            "audit_records": str(audit_records_path),
            "target_points": str(target_points_path),
            "gt_pose_dir": str(gt_pose_dir),
        },
        "config": {
            "point_stride": value(args, config, "point-stride", None),
            "fscore_thresholds": value(args, config, "fscore-thresholds", [0.02, 0.05, 0.1]),
            "coverage_radius": value(args, config, "coverage-radius", 0.05),
            "voxel_downsample_size": value(args, config, "voxel-downsample-size", 0.02),
            "max_metric_points": value(args, config, "max-metric-points", 12000),
            "metric_sample_method": value(args, config, "metric-sample-method", "random"),
            "seed": value(args, config, "seed", 0),
            "max_camera_anchor_condition_number": value(args, config, "max-camera-anchor-condition-number", 100.0),
        },
        "attempted_record_count": len(records),
        "did_run_vggt": False,
        "runtime_seconds": None,
    }

    invalid_caches = collect_invalid_reconstruction_caches(records)
    if invalid_caches:
        report = {
            **common_report,
            "status": "blocked_invalid_reconstruction_cache",
            "invalid_reconstruction_caches": invalid_caches,
            "missing_pose_enc_count": count_missing_pose_enc(invalid_caches),
            "required_next_action": "Regenerate oracle reconstruction caches as complete oracle-reconstruction-v3 artifacts after user approval; do not use candidate cameras as alignment anchors.",
        }
        report["runtime_seconds"] = time.perf_counter() - started_at
        write_json_report(output_dir / "stage_b_camera_anchor_report.json", report)
        write_markdown_report(output_dir / "run_log.md", report)
        print(json.dumps({
            "status": report["status"],
            "attempted_record_count": len(records),
            "invalid_reconstruction_cache_count": len(invalid_caches),
            "missing_pose_enc_count": report["missing_pose_enc_count"],
            "report": str(output_dir / "stage_b_camera_anchor_report.json"),
        }, indent=2))
        return

    point_stride_raw = value(args, config, "point-stride", None)
    point_stride = None if point_stride_raw is None else int(point_stride_raw)
    thresholds = tuple(float(item) for item in value(args, config, "fscore-thresholds", [0.02, 0.05, 0.1]))
    coverage_radius = float(value(args, config, "coverage-radius", 0.05))
    voxel_size_raw = value(args, config, "voxel-downsample-size", 0.02)
    voxel_size = None if voxel_size_raw is None else float(voxel_size_raw)
    max_metric_points = optional_int(value(args, config, "max-metric-points", 12000))
    metric_sample_method = str(value(args, config, "metric-sample-method", "random"))
    seed = int(value(args, config, "seed", 0))
    max_camera_anchor_condition_number = float(
        value(args, config, "max-camera-anchor-condition-number", 100.0)
    )

    try:
        output_records = compute_camera_anchor_records(
            records=records,
            target_points_path=target_points_path,
            gt_pose_dir=gt_pose_dir,
            point_stride=point_stride,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            voxel_size=voxel_size,
            max_metric_points=max_metric_points,
            seed=seed,
            metric_sample_method=metric_sample_method,
            max_camera_anchor_condition_number=max_camera_anchor_condition_number,
        )
    except DegenerateCameraAnchorsError as exc:
        report = {
            **common_report,
            "status": "blocked_degenerate_camera_anchors",
            "error": str(exc),
            "required_next_action": "Use non-degenerate observed camera anchors; do not add candidate cameras as anchors.",
        }
        report["runtime_seconds"] = time.perf_counter() - started_at
        write_json_report(output_dir / "stage_b_camera_anchor_report.json", report)
        write_markdown_report(output_dir / "run_log.md", report)
        print(json.dumps({
            "status": report["status"],
            "error": report["error"],
            "report": str(output_dir / "stage_b_camera_anchor_report.json"),
        }, indent=2))
        return

    records_path = output_dir / "camera_anchor_oracle_gain.jsonl"
    write_jsonl(records_path, output_records)
    summary = summarize_oracle_audit(output_records, observed_view_ids=output_records[0].observed_view_ids)
    summary_path = output_dir / "camera_anchor_summary.json"
    write_json_report(summary_path, summary)
    report = {
        **common_report,
        "status": "complete",
        "camera_anchor_summary": summary,
        "point_cloud_diagnostic_summary": summarize_point_cloud_diagnostics(output_records),
        "outputs": {
            "camera_anchor_records": str(records_path),
            "camera_anchor_summary": str(summary_path),
        },
    }
    report["runtime_seconds"] = time.perf_counter() - started_at
    write_json_report(output_dir / "stage_b_camera_anchor_report.json", report)
    write_markdown_report(output_dir / "run_log.md", report)
    print(json.dumps({
        "status": "complete",
        "records": len(output_records),
        "summary": str(summary_path),
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
