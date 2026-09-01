"""Stage-A audit for oracle metric/alignment stability using cached points only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt"
for import_root in (PROJECT_ROOT, VGGT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from viewtoken.oracle import (  # noqa: E402
    evaluate_alignment_and_metrics,
    flatten_gains,
    flatten_metrics,
    identical_cloud_gain_check,
    load_point_cloud,
    summarize_metric_stability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source-points", type=Path, default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--candidate-records", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    parser.add_argument("--alignment", choices=("identity", "sim3_icp"), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--fscore-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--coverage-radius", type=float, default=None)
    parser.add_argument("--voxel-downsample-size", type=float, default=None)
    parser.add_argument("--max-metric-points", type=int, default=None)
    parser.add_argument("--trim-fraction", type=float, default=None)
    parser.add_argument("--inlier-threshold", type=float, default=None)
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


def resolve_path(raw: Any, label: str) -> Path:
    if raw is None:
        raise ValueError(f"{label} is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_output_dir(raw: Any) -> Path:
    path = Path(raw or "outputs/oracle_calibration/stage_a_metric_stability").expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def flatten_record_gains(record: dict[str, Any]) -> dict[str, float]:
    gains = record["gains"]
    flat = {
        "chamfer": float(gains["chamfer"]),
        "accuracy": float(gains["accuracy"]),
        "completeness": float(gains["completeness"]),
        "coverage": float(gains["coverage"]),
    }
    for threshold, metric_value in gains["fscore"].items():
        flat[f"fscore@{threshold}"] = float(metric_value)
    return flat


def summarize_existing_candidate_records(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        return {"record_count": 0, "metrics": {}}
    observed = set(records[0].get("observed_view_ids", []))
    held_out = [record for record in records if record.get("candidate_view_id") not in observed]
    metric_names = sorted({name for record in held_out for name in flatten_record_gains(record)})
    metric_summary = {}
    for metric_name in metric_names:
        values = [flatten_record_gains(record)[metric_name] for record in held_out]
        metric_summary[metric_name] = {
            "candidate_count": len(values),
            "min": min(values),
            "median": median(values),
            "mean": mean(values),
            "std": pstdev(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
    return {
        "record_count": len(records),
        "held_out_record_count": len(held_out),
        "metrics": metric_summary,
    }


def add_noise_vs_candidate_spread(
    stability: dict[str, Any], candidate_summary: dict[str, Any] | None
) -> dict[str, Any]:
    if candidate_summary is None:
        return {}
    comparisons = {}
    for metric_name, metric_stability in stability["metrics"].items():
        candidate_metric = candidate_summary["metrics"].get(metric_name)
        if not candidate_metric:
            continue
        noise_std = metric_stability["std"]
        candidate_range = candidate_metric["range"]
        comparisons[metric_name] = {
            "multi_seed_metric_std": noise_std,
            "candidate_gain_range": candidate_range,
            "std_to_candidate_range_ratio": None
            if candidate_range == 0 or noise_std is None
            else noise_std / candidate_range,
        }
    return comparisons


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    stability = report["multi_seed_stability"]
    candidate = report.get("existing_candidate_gain_summary")
    comparisons = report.get("noise_vs_candidate_spread", {})

    metric_rows = []
    for metric_name, stats in stability["metrics"].items():
        metric_rows.append(
            f"| `{metric_name}` | `{stats['mean']:.8f}` | `{stats['std']:.8f}` | "
            f"`{stats['min']:.8f}` | `{stats['max']:.8f}` |"
        )

    alignment_rows = []
    for name, stats in stability.get("alignment", {}).items():
        alignment_rows.append(
            f"| `{name}` | `{stats['mean']:.8f}` | `{stats['std']:.8f}` | "
            f"`{stats['min']:.8f}` | `{stats['max']:.8f}` |"
        )

    comparison_rows = []
    for metric_name, stats in comparisons.items():
        ratio = stats["std_to_candidate_range_ratio"]
        ratio_text = "null" if ratio is None else f"{ratio:.8f}"
        comparison_rows.append(
            f"| `{metric_name}` | `{stats['multi_seed_metric_std']:.8f}` | "
            f"`{stats['candidate_gain_range']:.8f}` | `{ratio_text}` |"
        )

    text = f"""# Stage A Oracle Metric Stability Audit

This report uses cached reconstruction points only. It does not run VGGT or create new reconstructions.

## Inputs

- Source points: `{report['inputs']['source_points']}`
- Target points: `{report['inputs']['target_points']}`
- Alignment: `{report['config']['alignment']}`
- Seeds: `{report['config']['seeds']}`
- Voxel downsample size: `{report['config']['voxel_downsample_size']}`
- Max metric points: `{report['config']['max_metric_points']}`
- Trim fraction: `{report['config']['trim_fraction']}`
- Inlier threshold: `{report['config']['inlier_threshold']}`

## Identical-Cloud Check

- Max absolute gain: `{report['identical_cloud_check']['max_abs_gain']:.12g}`
- Gains: `{json.dumps(report['identical_cloud_check']['gains'], sort_keys=True)}`

## Multi-Seed Metric Stability

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## Multi-Seed Alignment Diagnostics

| diagnostic | mean | std | min | max |
|---|---:|---:|---:|---:|
{chr(10).join(alignment_rows)}
"""
    if candidate is not None:
        text += f"""
## Existing Candidate Gain Spread

Existing records: `{candidate['record_count']}`, held-out: `{candidate['held_out_record_count']}`.

| metric | metric std across seeds | candidate gain range | ratio |
|---|---:|---:|---:|
{chr(10).join(comparison_rows)}
"""
    text += f"""
## Interpretation

The identical-cloud check should be zero up to floating-point noise. Multi-seed standard deviations should be much smaller than real candidate gain differences before the oracle labels can be trusted.

Runtime seconds: `{report['runtime_seconds']:.3f}`
"""
    path.write_text(text)


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)

    source_points_path = resolve_path(value(args, config, "source-points", None), "source-points")
    target_points_path = resolve_path(value(args, config, "target-points", None), "target-points")
    candidate_records_raw = value(args, config, "candidate-records", None)
    candidate_records_path = None if candidate_records_raw is None else resolve_path(candidate_records_raw, "candidate-records")
    output_dir = resolve_output_dir(value(args, config, "output-dir", None))
    output_dir.mkdir(parents=True, exist_ok=True)

    point_stride = value(args, config, "point-stride", None)
    point_stride = None if point_stride is None else int(point_stride)
    alignment = str(value(args, config, "alignment", "sim3_icp"))
    seeds = [int(seed) for seed in value(args, config, "seeds", list(range(10)))]
    thresholds = tuple(float(item) for item in value(args, config, "fscore-thresholds", [0.02, 0.05, 0.1]))
    coverage_radius = float(value(args, config, "coverage-radius", 0.05))
    voxel_size_raw = value(args, config, "voxel-downsample-size", 0.02)
    voxel_size = None if voxel_size_raw is None else float(voxel_size_raw)
    max_metric_points_raw = value(args, config, "max-metric-points", 12000)
    max_metric_points = None if max_metric_points_raw is None else int(max_metric_points_raw)
    trim_fraction = float(value(args, config, "trim-fraction", 1.0))
    inlier_threshold_raw = value(args, config, "inlier-threshold", coverage_radius)
    inlier_threshold = None if inlier_threshold_raw is None else float(inlier_threshold_raw)

    source_points = load_point_cloud(source_points_path)
    target_points = load_point_cloud(target_points_path, point_stride=point_stride)

    identical = identical_cloud_gain_check(
        points=source_points,
        target=target_points,
        alignment=alignment,
        seed=seeds[0],
        thresholds=thresholds,
        coverage_radius=coverage_radius,
        max_points=max_metric_points,
        voxel_size=voxel_size,
        trim_fraction=trim_fraction,
        inlier_threshold=inlier_threshold,
    )
    results = [
        evaluate_alignment_and_metrics(
            points=source_points,
            target=target_points,
            alignment=alignment,
            seed=seed,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            trim_fraction=trim_fraction,
            inlier_threshold=inlier_threshold,
        )
        for seed in seeds
    ]
    stability = summarize_metric_stability(results)
    candidate_summary = summarize_existing_candidate_records(candidate_records_path)
    report = {
        "inputs": {
            "source_points": str(source_points_path),
            "target_points": str(target_points_path),
            "candidate_records": None if candidate_records_path is None else str(candidate_records_path),
            "source_point_count": int(source_points.shape[0]),
            "target_point_count": int(target_points.shape[0]),
        },
        "config": {
            "alignment": alignment,
            "seeds": seeds,
            "fscore_thresholds": list(thresholds),
            "coverage_radius": coverage_radius,
            "voxel_downsample_size": voxel_size,
            "max_metric_points": max_metric_points,
            "trim_fraction": trim_fraction,
            "inlier_threshold": inlier_threshold,
            "point_stride": point_stride,
        },
        "identical_cloud_check": identical,
        "per_seed_results": [result.to_dict() for result in results],
        "multi_seed_stability": stability,
        "existing_candidate_gain_summary": candidate_summary,
        "noise_vs_candidate_spread": add_noise_vs_candidate_spread(stability, candidate_summary),
        "runtime_seconds": time.perf_counter() - started_at,
    }

    json_path = output_dir / "stage_a_metric_stability.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    md_path = output_dir / "run_log.md"
    write_markdown_report(md_path, report)

    print(f"Wrote Stage-A metric stability report to {json_path}")
    print(json.dumps({
        "identical_max_abs_gain": report["identical_cloud_check"]["max_abs_gain"],
        "metric_stability": report["multi_seed_stability"]["metrics"],
        "alignment_stability": report["multi_seed_stability"].get("alignment", {}),
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
