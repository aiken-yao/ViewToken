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
    OracleGainRecord,
    PointCloudMetrics,
    compute_pointcloud_metrics,
    estimate_camera_anchor_alignment,
    load_gt_camera_centers_by_view_id,
    load_point_cloud,
    load_reconstruction_camera_centers,
    summarize_oracle_audit,
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
    parser.add_argument("--seed", type=int, default=None)
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


def collect_missing_pose_enc(records: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for record in records:
        for path_value in record.get("reconstruction_paths", {}).values():
            reconstruction_dir = Path(path_value).expanduser().resolve().parent
            pose_path = reconstruction_dir / "pose_enc.pt"
            if str(pose_path) not in seen and not pose_path.is_file():
                missing.append(str(pose_path))
                seen.add(str(pose_path))
    return missing


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    if report["status"] == "blocked_missing_pose_enc":
        missing = report["missing_pose_enc_paths"]
        missing_preview = "\n".join(f"- `{item}`" for item in missing[:12])
        if len(missing) > 12:
            missing_preview += f"\n- ... {len(missing) - 12} more"
        text = f"""# Stage B Camera-Anchor Alignment Audit

Status: `blocked_missing_pose_enc`

This Stage-B audit did not run VGGT and did not create new reconstructions. It attempted to reuse the existing audit20 cached reconstructions, but those caches do not contain `pose_enc.pt`, which is required to decode VGGT-predicted camera centers.

## Inputs

- Audit records: `{report['inputs']['audit_records']}`
- Target points: `{report['inputs']['target_points']}`
- GT pose dir: `{report['inputs']['gt_pose_dir']}`
- Attempted records: `{report['attempted_record_count']}`
- Missing reconstruction pose files: `{len(missing)}`

## Missing Files

{missing_preview}

## Interpretation

Stage B requires each baseline and observed-plus-candidate reconstruction cache to store VGGT `pose_enc.pt`. The existing audit20 cache predates that requirement and only stores `points.pt`, `confidence.pt`, and `metadata.json`. Under the current tips, the correct behavior is to stop here rather than rerun VGGT silently.

Future reconstruction caches now write `pose_enc.pt` and use cache schema `oracle-reconstruction-v3`; after the user approves regenerating caches, this script can compute camera-anchored metrics without using candidate cameras as alignment anchors.
"""
    else:
        held = report["camera_anchor_summary"]["held_out_metrics"]
        rows = []
        for metric_name, stats in held.items():
            rows.append(
                f"| `{metric_name}` | `{stats['mean']:.8f}` | `{stats['positive_gain_ratio']:.3f}` | "
                f"`{stats['oracle_best_candidate']}` | `{stats['oracle_best_gain']:.8f}` |"
            )
        text = f"""# Stage B Camera-Anchor Alignment Audit

Status: `complete`

This audit reuses cached reconstructions and aligns each reconstruction to ScanNet GT using only the shared observed camera centers. Candidate cameras are not used as alignment anchors.

| metric | mean gain | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---|---:|
{chr(10).join(rows)}

Output records: `{report['outputs']['camera_anchor_records']}`
Summary JSON: `{report['outputs']['camera_anchor_summary']}`
"""
    path.write_text(text)


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
        )
        candidate_alignment = estimate_camera_anchor_alignment(
            candidate_centers,
            gt_centers,
            shared_anchor_ids=observed_ids,
        )

        baseline_points = load_point_cloud(baseline_points_path)
        candidate_points = load_point_cloud(candidate_points_path)
        baseline_metrics = compute_pointcloud_metrics(
            baseline_alignment.transform.apply(baseline_points),
            target_points,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
        )
        candidate_metrics = compute_pointcloud_metrics(
            candidate_alignment.transform.apply(candidate_points),
            target_points,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=seed,
        )

        metadata = dict(record.get("metadata", {}))
        metadata["alignment_protocol"] = "camera_anchor_sim3_shared_observed"
        metadata["candidate_used_as_alignment_anchor"] = False
        metadata["shared_anchor_ids"] = observed_ids
        metadata["baseline_camera_anchor_alignment"] = baseline_alignment.to_dict()
        metadata["candidate_camera_anchor_alignment"] = candidate_alignment.to_dict()
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
            "seed": value(args, config, "seed", 0),
        },
        "attempted_record_count": len(records),
        "did_run_vggt": False,
        "runtime_seconds": None,
    }

    missing_pose_enc = collect_missing_pose_enc(records)
    if missing_pose_enc:
        report = {
            **common_report,
            "status": "blocked_missing_pose_enc",
            "missing_pose_enc_paths": missing_pose_enc,
            "required_next_action": "Regenerate oracle reconstruction caches with pose_enc.pt after user approval; do not use candidate cameras as alignment anchors.",
        }
        report["runtime_seconds"] = time.perf_counter() - started_at
        write_json_report(output_dir / "stage_b_camera_anchor_report.json", report)
        write_markdown_report(output_dir / "run_log.md", report)
        print(json.dumps({
            "status": report["status"],
            "attempted_record_count": len(records),
            "missing_pose_enc_count": len(missing_pose_enc),
            "report": str(output_dir / "stage_b_camera_anchor_report.json"),
        }, indent=2))
        return

    point_stride_raw = value(args, config, "point-stride", None)
    point_stride = None if point_stride_raw is None else int(point_stride_raw)
    thresholds = tuple(float(item) for item in value(args, config, "fscore-thresholds", [0.02, 0.05, 0.1]))
    coverage_radius = float(value(args, config, "coverage-radius", 0.05))
    voxel_size_raw = value(args, config, "voxel-downsample-size", 0.02)
    voxel_size = None if voxel_size_raw is None else float(voxel_size_raw)
    max_metric_points_raw = value(args, config, "max-metric-points", 12000)
    max_metric_points = None if max_metric_points_raw is None else int(max_metric_points_raw)
    seed = int(value(args, config, "seed", 0))

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
    )
    records_path = output_dir / "camera_anchor_oracle_gain.jsonl"
    write_jsonl(records_path, output_records)
    summary = summarize_oracle_audit(output_records, observed_view_ids=output_records[0].observed_view_ids)
    summary_path = output_dir / "camera_anchor_summary.json"
    write_json_report(summary_path, summary)
    report = {
        **common_report,
        "status": "complete",
        "camera_anchor_summary": summary,
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
