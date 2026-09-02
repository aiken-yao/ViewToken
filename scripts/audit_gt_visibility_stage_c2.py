"""Stage-C2 GT visibility and novel-surface audit using cached reconstructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewtoken.oracle.cache import (  # noqa: E402
    ReconstructionCacheValidationError,
    validate_reconstruction_cache,
)
from viewtoken.oracle.io import load_point_cloud, load_pose_matrix  # noqa: E402
from viewtoken.oracle.metrics import (  # noqa: E402
    SimilarityTransform,
    nearest_neighbor_squared_distances,
    sample_points,
    voxel_downsample_points,
)
from viewtoken.oracle.visibility import (  # noqa: E402
    PinholeIntrinsics,
    build_visibility_masks,
    camera_pose_delta_to_observed,
    summarize_visibility_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audit-records", type=Path, default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--posed-image-dir", type=Path, default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return payload


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def load_pose(path: Path) -> torch.Tensor:
    return torch.tensor(load_pose_matrix(path), dtype=torch.float32)


def load_intrinsics(path: Path, image_size_wh: tuple[int, int]) -> PinholeIntrinsics:
    matrix = torch.tensor(np.loadtxt(path, dtype=np.float64), dtype=torch.float64)
    return PinholeIntrinsics.from_matrix(
        matrix,
        width=int(image_size_wh[0]),
        height=int(image_size_wh[1]),
    )


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
            try:
                validate_reconstruction_cache(
                    reconstruction_dir,
                    expected_fingerprint=expected_reconstruction_fingerprint(record, role),
                )
            except ReconstructionCacheValidationError as exc:
                invalid.append(
                    {
                        "record_index": record_index,
                        "role": role,
                        "reconstruction_dir": str(reconstruction_dir),
                        "errors": exc.errors,
                    }
                )
    return invalid


def sorted_view_ids(records: list[dict[str, Any]]) -> list[str]:
    view_ids: set[str] = set()
    for record in records:
        view_ids.update(str(view_id) for view_id in record.get("observed_view_ids", []))
        view_ids.add(str(record["candidate_view_id"]))
    return sorted(view_ids)


def inspect_assets(
    records: list[dict[str, Any]],
    target_points: Path,
    posed_image_dir: Path,
    intrinsics_path: Path,
    image_extension: str,
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    view_ids = sorted_view_ids(records)
    image_sizes: dict[str, list[int]] = {}
    if not posed_image_dir.is_dir():
        missing.append(f"posed_image_dir:{posed_image_dir}")
    for view_id in view_ids:
        pose_path = posed_image_dir / f"{view_id}.txt"
        image_path = posed_image_dir / f"{view_id}.{image_extension.lstrip('.')}"
        if not pose_path.is_file():
            missing.append(f"pose:{pose_path}")
        if not image_path.is_file():
            missing.append(f"rgb:{image_path}")
        else:
            image_sizes[view_id] = list(read_image_size(image_path))
    if not target_points.is_file():
        missing.append(f"target_points:{target_points}")
    if not intrinsics_path.is_file():
        missing.append(f"intrinsics:{intrinsics_path}")

    depth_like = sorted(
        [
            str(path)
            for path in posed_image_dir.glob("*")
            if path.is_file()
            and (
                "depth" in path.name.lower()
                or path.suffix.lower() in {".png", ".npy", ".npz"}
            )
        ]
    )
    asset_report = {
        "target_points": str(target_points),
        "target_points_size_bytes": target_points.stat().st_size if target_points.is_file() else None,
        "posed_image_dir": str(posed_image_dir),
        "intrinsics": str(intrinsics_path),
        "view_ids_checked": view_ids,
        "image_extension": image_extension,
        "image_sizes_wh": image_sizes,
        "depth_like_files_found": depth_like,
        "depth_maps_found": bool(depth_like),
    }
    return asset_report, missing


def sim3_from_dict(payload: dict[str, Any]) -> SimilarityTransform:
    return SimilarityTransform(
        scale=float(payload["scale"]),
        rotation=torch.tensor(payload["rotation"], dtype=torch.float32),
        translation=torch.tensor(payload["translation"], dtype=torch.float32),
    )


def alignment_transform(record: dict[str, Any], role: str) -> SimilarityTransform:
    key = f"{role}_camera_anchor_alignment"
    return sim3_from_dict(record["metadata"][key]["transform"])


def load_aligned_points(record: dict[str, Any], role: str) -> torch.Tensor:
    points_path = Path(record["reconstruction_paths"][role]).expanduser().resolve()
    transform = alignment_transform(record, role)
    return transform.apply(load_point_cloud(points_path))


def point_fingerprint(points: torch.Tensor) -> str:
    quantized = torch.round(points.float().cpu() * 10000.0).to(torch.int64).contiguous()
    return hashlib.sha1(quantized.numpy().tobytes()).hexdigest()


def prepare_metric_points(
    points: torch.Tensor,
    voxel_size: float | None,
    max_points: int | None,
    seed: int,
    method: str,
) -> torch.Tensor:
    downsampled = voxel_downsample_points(points, voxel_size)
    return sample_points(downsampled, max_points=max_points, seed=seed, method=method).float().cpu()


def nearest_distances(source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return nearest_neighbor_squared_distances(source, target, chunk_size=chunk_size).sqrt()


def masked_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def masked_mean(distances: torch.Tensor, mask: torch.Tensor) -> float | None:
    if masked_count(mask) == 0:
        return None
    return float(distances[mask].mean().item())


def coverage_ratio(distances: torch.Tensor, mask: torch.Tensor, threshold: float) -> float | None:
    if masked_count(mask) == 0:
        return None
    return float((distances[mask] <= float(threshold)).float().mean().item())


def covered_count(distances: torch.Tensor, mask: torch.Tensor, threshold: float) -> int:
    if masked_count(mask) == 0:
        return 0
    return int((distances[mask] <= float(threshold)).sum().item())


def subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def summarize_metric_values(values: list[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "range": None}
    tensor = torch.tensor(finite, dtype=torch.float64)
    std = 0.0 if tensor.numel() == 1 else float(tensor.std(unbiased=False).item())
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    return {
        "count": len(finite),
        "mean": float(tensor.mean().item()),
        "std": std,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
    }


def connected_component_summary(
    points: torch.Tensor,
    mask: torch.Tensor,
    radius: float,
) -> dict[str, Any]:
    subset = points[mask].float().cpu()
    count = int(subset.shape[0])
    if count == 0:
        return {
            "component_radius_meters": radius,
            "point_count": 0,
            "component_count": 0,
            "largest_component_count": 0,
            "largest_component_fraction_of_novel": None,
        }

    parent = list(range(count))
    rank = [0] * count

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    cell_size = float(radius)
    cells: dict[tuple[int, int, int], list[int]] = {}
    quantized = torch.floor(subset / cell_size).to(torch.int64)
    offsets = list(product((-1, 0, 1), repeat=3))
    radius_squared = float(radius) * float(radius)
    for index in range(count):
        cell = tuple(int(value) for value in quantized[index].tolist())
        for offset in offsets:
            neighbor_cell = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            for other in cells.get(neighbor_cell, []):
                if torch.sum((subset[index] - subset[other]).square()).item() <= radius_squared:
                    union(index, other)
        cells.setdefault(cell, []).append(index)

    component_sizes = Counter(find(index) for index in range(count))
    largest = max(component_sizes.values())
    return {
        "component_radius_meters": radius,
        "point_count": count,
        "component_count": len(component_sizes),
        "largest_component_count": int(largest),
        "largest_component_fraction_of_novel": float(largest) / float(count),
    }


def compute_visibility_metrics(
    target_distances_to_baseline: torch.Tensor,
    target_distances_to_candidate: torch.Tensor,
    baseline_distances_to_target: torch.Tensor,
    candidate_distances_to_target: torch.Tensor,
    masks: Any,
    coverage_thresholds: tuple[float, ...],
    outlier_threshold: float,
) -> dict[str, Any]:
    total_count = int(masks.observed.numel())
    metrics: dict[str, Any] = {
        "novel_coverage_gain": {},
        "novel_surface_gain_scene_normalized": {},
        "observed_retention_gain": {},
        "visible_union_completeness_gain": subtract_optional(
            masked_mean(target_distances_to_baseline, masks.union),
            masked_mean(target_distances_to_candidate, masks.union),
        ),
        "global_accuracy": {
            "baseline": float(baseline_distances_to_target.mean().item()),
            "candidate": float(candidate_distances_to_target.mean().item()),
        },
        "global_outlier_ratio": {
            "threshold_meters": outlier_threshold,
            "baseline": float((baseline_distances_to_target > outlier_threshold).float().mean().item()),
            "candidate": float((candidate_distances_to_target > outlier_threshold).float().mean().item()),
        },
    }
    metrics["global_accuracy"]["gain"] = (
        metrics["global_accuracy"]["baseline"] - metrics["global_accuracy"]["candidate"]
    )
    metrics["global_outlier_ratio"]["gain"] = (
        metrics["global_outlier_ratio"]["baseline"] - metrics["global_outlier_ratio"]["candidate"]
    )

    for threshold in coverage_thresholds:
        key = f"{threshold:g}"
        baseline_novel = coverage_ratio(target_distances_to_baseline, masks.novel, threshold)
        candidate_novel = coverage_ratio(target_distances_to_candidate, masks.novel, threshold)
        metrics["novel_coverage_gain"][key] = {
            "baseline": baseline_novel,
            "candidate": candidate_novel,
            "gain": subtract_optional(candidate_novel, baseline_novel),
        }
        baseline_covered = covered_count(target_distances_to_baseline, masks.novel, threshold)
        candidate_covered = covered_count(target_distances_to_candidate, masks.novel, threshold)
        metrics["novel_surface_gain_scene_normalized"][key] = (
            float(candidate_covered - baseline_covered) / float(total_count)
        )

        baseline_observed = coverage_ratio(target_distances_to_baseline, masks.observed, threshold)
        candidate_observed = coverage_ratio(target_distances_to_candidate, masks.observed, threshold)
        metrics["observed_retention_gain"][key] = {
            "baseline": baseline_observed,
            "candidate": candidate_observed,
            "gain": subtract_optional(candidate_observed, baseline_observed),
        }
    return metrics


def old_global_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_metrics": record.get("baseline_metrics"),
        "candidate_metrics": record.get("candidate_metrics"),
        "gains": record.get("gains"),
    }


def scores_by_metric(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        payload: Any = row
        for part in metric_path:
            payload = payload[part]
        scored.append({"candidate_view_id": str(row["candidate_view_id"]), "score": float(payload)})
    return sorted(scored, key=lambda item: (-item["score"], item["candidate_view_id"]))


def best_candidate_tie_group(scored: list[dict[str, Any]], tolerance: float = 1e-12) -> list[str]:
    if not scored:
        return []
    best_score = float(scored[0]["score"])
    return [
        str(item["candidate_view_id"])
        for item in scored
        if abs(float(item["score"]) - best_score) <= tolerance
    ]


def build_assessment(
    nominal_rows: list[dict[str, Any]],
    stability_by_candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_candidate = {str(row["candidate_view_id"]): row for row in nominal_rows}
    duplicate = by_candidate.get("00010")
    high_overlap = by_candidate.get("00019")
    connected_candidates = [by_candidate[item] for item in ("00325", "00425") if item in by_candidate]

    duplicate_limit = float(config.get("duplicate-novel-scene-fraction-max", 0.005))
    retention_limit = float(config.get("observed-retention-gain-min", -0.02))
    connectivity_min_count = int(config.get("novel-connectivity-min-largest-component-count", 10))
    duplicate_ok = (
        duplicate is not None
        and duplicate["visibility_stats"]["novel_scene_fraction"] <= duplicate_limit
    )
    connected_with_novel = [
        row
        for row in connected_candidates
        if row["visibility_stats"]["novel_count"] > 0
        and row["novel_connectivity"]["largest_component_count"] >= connectivity_min_count
    ]
    high_overlap_novel_less_than_connected = False
    if high_overlap is not None and connected_with_novel:
        high_overlap_novel_less_than_connected = (
            high_overlap["visibility_stats"]["novel_scene_fraction"]
            < max(row["visibility_stats"]["novel_scene_fraction"] for row in connected_with_novel)
        )

    stable_positive = []
    for candidate_id in ("00325", "00425"):
        stats = stability_by_candidate.get(candidate_id, {})
        gain_005 = stats.get("novel_scene_normalized_gain_0.05", {}).get("mean")
        gain_010 = stats.get("novel_scene_normalized_gain_0.1", {}).get("mean")
        if gain_005 is not None and gain_010 is not None and gain_005 > 0.0 and gain_010 > 0.0:
            stable_positive.append(candidate_id)

    damaged_retention = []
    for row in nominal_rows:
        gains = row["visibility_metrics"]["observed_retention_gain"]
        for threshold, payload in gains.items():
            gain = payload["gain"]
            if gain is not None and gain < retention_limit:
                damaged_retention.append(
                    {
                        "candidate_view_id": row["candidate_view_id"],
                        "threshold": threshold,
                        "gain": gain,
                    }
                )

    passed = (
        duplicate_ok
        and high_overlap_novel_less_than_connected
        and bool(stable_positive)
        and not damaged_retention
    )
    return {
        "stage_c2_passed": passed,
        "duplicate_novel_near_zero": duplicate_ok,
        "duplicate_novel_scene_fraction_max": duplicate_limit,
        "high_overlap_novel_less_than_connected_new_area": high_overlap_novel_less_than_connected,
        "connected_new_area_candidates_with_novel_surface": [
            row["candidate_view_id"] for row in connected_with_novel
        ],
        "novel_connectivity_min_largest_component_count": connectivity_min_count,
        "stable_positive_connected_new_area_candidates": stable_positive,
        "observed_retention_not_badly_damaged": not damaged_retention,
        "observed_retention_gain_min": retention_limit,
        "damaged_observed_retention": damaged_retention,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    if summary["status"] == "blocked_missing_visibility_assets":
        missing = "\n".join(f"- `{item}`" for item in summary["missing_visibility_assets"])
        text = f"""# Stage C2 GT Visibility Audit

Status: `blocked_missing_visibility_assets`

The audit did not run VGGT and did not use camera-distance fallback. It stopped because required visibility assets are missing.

## Missing Assets

{missing}

## Inputs

- Audit records: `{summary['inputs']['audit_records']}`
- Target points: `{summary['inputs']['target_points']}`
- Posed image dir: `{summary['inputs']['posed_image_dir']}`
- Intrinsics: `{summary['inputs']['intrinsics']}`
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return

    rows = []
    for row in summary["nominal_candidate_rows"]:
        stats = row["visibility_stats"]
        metrics = row["visibility_metrics"]
        conn = row["novel_connectivity"]
        delta = row["camera_delta_to_observed"]
        rows.append(
            "| `{candidate}` | `{tags}` | `{visible:.6f}` | `{novel_frac:.6f}` | "
            "`{novel_scene:.6f}` | `{overlap:.6f}` | `{largest}` | `{gain05:.6f}` | "
            "`{gain10:.6f}` | `{ret05:.6f}` | `{ret10:.6f}` | `{union:.6f}` | "
            "`{acc:.6f}` | `{dist:.6f}` | `{angle:.3f}` |".format(
                candidate=row["candidate_view_id"],
                tags=",".join(row["candidate_sanity_tags"]),
                visible=stats["candidate_fraction"],
                novel_frac=stats["candidate_novel_fraction"] or 0.0,
                novel_scene=stats["novel_scene_fraction"],
                overlap=stats["candidate_overlap_fraction"] or 0.0,
                largest=conn["largest_component_count"],
                gain05=metrics["novel_surface_gain_scene_normalized"]["0.05"],
                gain10=metrics["novel_surface_gain_scene_normalized"]["0.1"],
                ret05=metrics["observed_retention_gain"]["0.05"]["gain"] or 0.0,
                ret10=metrics["observed_retention_gain"]["0.1"]["gain"] or 0.0,
                union=metrics["visible_union_completeness_gain"] or 0.0,
                acc=metrics["global_accuracy"]["gain"],
                dist=delta["min_distance_to_observed_meters"],
                angle=delta["min_view_direction_change_degrees"],
            )
        )

    stability_rows = []
    for candidate_id, stats in sorted(summary["stability_by_candidate"].items()):
        stability_rows.append(
            "| `{candidate}` | `{novel_mean:.6f}` | `{novel_std:.6f}` | `{gain05_mean:.6f}` | "
            "`{gain05_std:.6f}` | `{gain10_mean:.6f}` | `{gain10_std:.6f}` |".format(
                candidate=candidate_id,
                novel_mean=stats["novel_scene_fraction"]["mean"] or 0.0,
                novel_std=stats["novel_scene_fraction"]["std"] or 0.0,
                gain05_mean=stats["novel_scene_normalized_gain_0.05"]["mean"] or 0.0,
                gain05_std=stats["novel_scene_normalized_gain_0.05"]["std"] or 0.0,
                gain10_mean=stats["novel_scene_normalized_gain_0.1"]["mean"] or 0.0,
                gain10_std=stats["novel_scene_normalized_gain_0.1"]["std"] or 0.0,
            )
        )

    assessment = summary["assessment"]
    text = f"""# Stage C2 GT Visibility Audit

Status: `{summary['status']}`

Stage C2 passed: `{assessment['stage_c2_passed']}`

This audit reused the complete Stage C deterministic v3 reconstruction caches and did not run VGGT. Candidate RGB/depth visibility is used only for offline oracle-label audit, not as future policy input.

## Convention

- ScanNet pose convention: `{summary['conventions']['pose_convention']}`
- Projection convention: `{summary['conventions']['projection_convention']}`
- Image size: `{summary['conventions']['image_size_wh'][0]}x{summary['conventions']['image_size_wh'][1]}`
- Intrinsics source: `{summary['conventions']['intrinsics_source']}`
- Occlusion source: `{summary['conventions']['occlusion_source']}`
- Nominal voxel/depth/seed: `{summary['nominal_key']}`
- Synthetic visibility test count: `{summary['test_count']['synthetic_visibility_tests']}`

## Nominal Results

| cand | tags | visible scene frac | cand novel frac | novel scene frac | cand overlap frac | largest novel comp | novel gain@0.05 scene | novel gain@0.10 scene | obs retention@0.05 | obs retention@0.10 | union compl gain | global acc gain | min dist | min angle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Stability

| cand | novel scene mean | novel scene std | gain@0.05 mean | gain@0.05 std | gain@0.10 mean | gain@0.10 std |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stability_rows)}

## Assessment

- Duplicate novelty near zero: `{assessment['duplicate_novel_near_zero']}`
- High-overlap novelty less than connected new area: `{assessment['high_overlap_novel_less_than_connected_new_area']}`
- Connected new-area candidates with novel surface: `{assessment['connected_new_area_candidates_with_novel_surface']}`
- Novel connectivity min largest component count: `{assessment['novel_connectivity_min_largest_component_count']}`
- Stable positive connected new-area candidates: `{assessment['stable_positive_connected_new_area_candidates']}`
- Observed retention not badly damaged: `{assessment['observed_retention_not_badly_damaged']}`
- Strict best candidate counts by gain@0.05: `{summary['ranking_stability']['strict_best_candidate_counts_by_gain_0.05']}`
- Best-candidate tie groups by gain@0.05: `{summary['ranking_stability']['best_candidate_tie_counts_by_gain_0.05']}`

Full JSON summary: `{summary['outputs']['docs_summary_json']}`
Variant records: `{summary['outputs']['variant_records_jsonl']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    audit_records_path = resolve_path(value(args, config, "audit-records", None), "audit-records")
    target_points_path = resolve_path(value(args, config, "target-points", None), "target-points", must_exist=False)
    posed_image_dir = resolve_path(value(args, config, "posed-image-dir", None), "posed-image-dir", must_exist=False)
    intrinsics_path = resolve_path(value(args, config, "intrinsics", None), "intrinsics", must_exist=False)
    output_dir = resolve_path(
        value(args, config, "output-dir", "outputs/oracle_calibration/scannet_scene0000_00_stage_c2"),
        "output-dir",
        must_exist=False,
    )
    docs_output_dir = resolve_path(
        value(args, config, "docs-output-dir", "docs/audits/scannet_scene0000_00_stage_c2"),
        "docs-output-dir",
        must_exist=False,
    )

    records = load_jsonl(audit_records_path)
    candidate_filter = value(args, config, "candidate-view-ids", None)
    if candidate_filter is not None:
        keep = {str(item) for item in candidate_filter}
        records = [record for record in records if str(record["candidate_view_id"]) in keep]

    image_extension = str(value(args, config, "image-extension", "jpg"))
    asset_report, missing_assets = inspect_assets(
        records=records,
        target_points=target_points_path,
        posed_image_dir=posed_image_dir,
        intrinsics_path=intrinsics_path,
        image_extension=image_extension,
    )
    invalid_caches = collect_invalid_reconstruction_caches(records)
    if invalid_caches:
        missing_assets.extend(
            f"invalid_reconstruction_cache:{item['reconstruction_dir']}:{';'.join(item['errors'])}"
            for item in invalid_caches
        )

    common_summary = {
        "inputs": {
            "audit_records": str(audit_records_path),
            "target_points": str(target_points_path),
            "posed_image_dir": str(posed_image_dir),
            "intrinsics": str(intrinsics_path),
        },
        "asset_report": asset_report,
        "did_run_vggt": False,
        "policy_inputs_exclude_candidate_rgb_depth_features_visibility": True,
        "runtime_seconds": None,
    }
    if missing_assets:
        summary = {
            **common_summary,
            "status": "blocked_missing_visibility_assets",
            "missing_visibility_assets": missing_assets,
            "required_next_action": "Provide exact missing GT visibility assets; do not fall back to camera distance only.",
        }
        summary["runtime_seconds"] = time.perf_counter() - started_at
        write_json(output_dir / "stage_c2_visibility_summary.json", summary)
        write_json(docs_output_dir / "stage_c2_visibility_summary.json", summary)
        write_markdown(docs_output_dir / "run_log.md", summary)
        print(json.dumps({"status": summary["status"], "missing": missing_assets}, indent=2))
        return

    observed_ids = list(records[0]["observed_view_ids"])
    all_image_sizes = list(asset_report["image_sizes_wh"].values())
    image_size_wh = tuple(all_image_sizes[0])
    if any(tuple(size) != image_size_wh for size in all_image_sizes):
        raise ValueError(f"Mixed image sizes are not supported: {asset_report['image_sizes_wh']}")
    intrinsics = load_intrinsics(intrinsics_path, image_size_wh=image_size_wh)
    observed_poses = {
        view_id: load_pose(posed_image_dir / f"{view_id}.txt")
        for view_id in observed_ids
    }
    candidate_poses = {
        str(record["candidate_view_id"]): load_pose(posed_image_dir / f"{record['candidate_view_id']}.txt")
        for record in records
    }

    target_points = load_point_cloud(
        target_points_path,
        point_stride=int(value(args, config, "point-stride", 6)),
    )
    baseline_aligned = load_aligned_points(records[0], "baseline")

    voxel_sizes = [float(item) for item in value(args, config, "gt-voxel-sizes", [0.02])]
    depth_tolerances = [float(item) for item in value(args, config, "visibility-depth-tolerances", [0.05])]
    seed_offsets = [int(item) for item in value(args, config, "surface-sample-seed-offsets", [0])]
    seed = int(value(args, config, "seed", 0))
    max_surface_points = int(value(args, config, "max-surface-points", 12000))
    max_prediction_points = int(value(args, config, "max-prediction-points", 12000))
    sample_method = str(value(args, config, "surface-sample-method", "hash"))
    prediction_sample_method = str(value(args, config, "prediction-sample-method", sample_method))
    coverage_thresholds = tuple(float(item) for item in value(args, config, "coverage-thresholds", [0.05, 0.1]))
    outlier_threshold = float(value(args, config, "global-outlier-threshold", 0.1))
    chunk_size = int(value(args, config, "nn-chunk-size", 512))
    connectivity_radius = float(value(args, config, "novel-connectivity-radius", 0.08))
    nominal_key = {
        "gt_voxel_size": float(value(args, config, "nominal-gt-voxel-size", 0.02)),
        "visibility_depth_tolerance": float(value(args, config, "nominal-visibility-depth-tolerance", 0.05)),
        "surface_sample_seed_offset": int(value(args, config, "nominal-surface-sample-seed-offset", 0)),
    }

    target_samples: dict[tuple[float, int], dict[str, Any]] = {}
    baseline_samples: dict[tuple[float, int], dict[str, torch.Tensor]] = {}

    def target_sample(voxel_size: float, seed_offset: int) -> dict[str, Any]:
        key = (voxel_size, seed_offset)
        if key not in target_samples:
            sample = prepare_metric_points(
                target_points,
                voxel_size=voxel_size,
                max_points=max_surface_points,
                seed=seed + seed_offset,
                method=sample_method,
            )
            target_samples[key] = {
                "points": sample,
                "fingerprint": point_fingerprint(sample),
            }
        return target_samples[key]

    def baseline_sample(voxel_size: float, seed_offset: int, target: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (voxel_size, seed_offset)
        if key not in baseline_samples:
            sample = prepare_metric_points(
                baseline_aligned,
                voxel_size=voxel_size,
                max_points=max_prediction_points,
                seed=seed + seed_offset + 101,
                method=prediction_sample_method,
            )
            baseline_samples[key] = {
                "points": sample,
                "target_to_prediction": nearest_distances(target, sample, chunk_size=chunk_size),
                "prediction_to_target": nearest_distances(sample, target, chunk_size=chunk_size),
            }
        return baseline_samples[key]

    variant_records: list[dict[str, Any]] = []
    nominal_candidate_rows: list[dict[str, Any]] = []
    rows_for_ranking_by_variant: dict[tuple[float, float, int], list[dict[str, Any]]] = {}

    for record in records:
        candidate_id = str(record["candidate_view_id"])
        candidate_aligned = load_aligned_points(record, "candidate")
        for voxel_size in voxel_sizes:
            for seed_offset in seed_offsets:
                target_payload = target_sample(voxel_size, seed_offset)
                surface_points = target_payload["points"]
                baseline_payload = baseline_sample(voxel_size, seed_offset, surface_points)
                candidate_sample = prepare_metric_points(
                    candidate_aligned,
                    voxel_size=voxel_size,
                    max_points=max_prediction_points,
                    seed=seed + seed_offset + 211,
                    method=prediction_sample_method,
                )
                target_to_candidate = nearest_distances(
                    surface_points,
                    candidate_sample,
                    chunk_size=chunk_size,
                )
                candidate_to_target = nearest_distances(
                    candidate_sample,
                    surface_points,
                    chunk_size=chunk_size,
                )
                for depth_tolerance in depth_tolerances:
                    masks = build_visibility_masks(
                        surface_points,
                        observed_poses.values(),
                        candidate_poses[candidate_id],
                        intrinsics=intrinsics,
                        depth_tolerance=depth_tolerance,
                    )
                    stats = summarize_visibility_masks(masks)
                    metrics = compute_visibility_metrics(
                        target_distances_to_baseline=baseline_payload["target_to_prediction"],
                        target_distances_to_candidate=target_to_candidate,
                        baseline_distances_to_target=baseline_payload["prediction_to_target"],
                        candidate_distances_to_target=candidate_to_target,
                        masks=masks,
                        coverage_thresholds=coverage_thresholds,
                        outlier_threshold=outlier_threshold,
                    )
                    variant_key = {
                        "gt_voxel_size": voxel_size,
                        "visibility_depth_tolerance": depth_tolerance,
                        "surface_sample_seed_offset": seed_offset,
                    }
                    row = {
                        "scene_id": record["scene_id"],
                        "candidate_view_id": candidate_id,
                        "candidate_sanity_tags": record.get("metadata", {}).get("candidate_sanity_tags", []),
                        "observed_view_ids": observed_ids,
                        "variant_key": variant_key,
                        "surface_sample": {
                            "count": int(surface_points.shape[0]),
                            "fingerprint": target_payload["fingerprint"],
                            "method": sample_method,
                            "max_points": max_surface_points,
                        },
                        "visibility_stats": stats.to_dict(),
                        "visibility_metrics": metrics,
                        "camera_delta_to_observed": camera_pose_delta_to_observed(
                            candidate_poses[candidate_id],
                            observed_poses,
                        ).to_dict(),
                        "old_global_diagnostics": old_global_diagnostics(record),
                    }
                    if variant_key == nominal_key:
                        row["novel_connectivity"] = connected_component_summary(
                            surface_points,
                            masks.novel,
                            radius=connectivity_radius,
                        )
                        nominal_candidate_rows.append(row)
                    variant_records.append(row)
                    rows_for_ranking_by_variant.setdefault(
                        (voxel_size, depth_tolerance, seed_offset),
                        [],
                    ).append(row)

    stability_by_candidate: dict[str, Any] = {}
    for candidate_id in sorted({str(record["candidate_view_id"]) for record in records}):
        candidate_rows = [row for row in variant_records if row["candidate_view_id"] == candidate_id]
        stability_by_candidate[candidate_id] = {
            "variant_count": len(candidate_rows),
            "novel_scene_fraction": summarize_metric_values(
                [row["visibility_stats"]["novel_scene_fraction"] for row in candidate_rows]
            ),
            "candidate_novel_fraction": summarize_metric_values(
                [row["visibility_stats"]["candidate_novel_fraction"] for row in candidate_rows]
            ),
            "novel_scene_normalized_gain_0.05": summarize_metric_values(
                [
                    row["visibility_metrics"]["novel_surface_gain_scene_normalized"].get("0.05")
                    for row in candidate_rows
                ]
            ),
            "novel_scene_normalized_gain_0.1": summarize_metric_values(
                [
                    row["visibility_metrics"]["novel_surface_gain_scene_normalized"].get("0.1")
                    for row in candidate_rows
                ]
            ),
            "observed_retention_gain_0.05": summarize_metric_values(
                [
                    row["visibility_metrics"]["observed_retention_gain"].get("0.05", {}).get("gain")
                    for row in candidate_rows
                ]
            ),
            "observed_retention_gain_0.1": summarize_metric_values(
                [
                    row["visibility_metrics"]["observed_retention_gain"].get("0.1", {}).get("gain")
                    for row in candidate_rows
                ]
            ),
            "visible_union_completeness_gain": summarize_metric_values(
                [row["visibility_metrics"]["visible_union_completeness_gain"] for row in candidate_rows]
            ),
        }

    ranking_variants = []
    best_counts: Counter[str] = Counter()
    best_tie_counts: Counter[str] = Counter()
    for key, rows in sorted(rows_for_ranking_by_variant.items()):
        scored = scores_by_metric(
            rows,
            ("visibility_metrics", "novel_surface_gain_scene_normalized", "0.05"),
        )
        ordered = [str(item["candidate_view_id"]) for item in scored]
        best_ties = best_candidate_tie_group(scored)
        if len(best_ties) == 1:
            best_counts[best_ties[0]] += 1
        elif best_ties:
            best_tie_counts["|".join(best_ties)] += 1
        ranking_variants.append(
            {
                "variant_key": {
                    "gt_voxel_size": key[0],
                    "visibility_depth_tolerance": key[1],
                    "surface_sample_seed_offset": key[2],
                },
                "scores_by_novel_surface_gain_scene_normalized_0.05": scored,
                "ranking_by_novel_surface_gain_scene_normalized_0.05": ordered,
                "best_candidate_ties_by_novel_surface_gain_scene_normalized_0.05": best_ties,
            }
        )

    nominal_candidate_rows = sorted(nominal_candidate_rows, key=lambda row: row["candidate_view_id"])
    assessment = build_assessment(
        nominal_rows=nominal_candidate_rows,
        stability_by_candidate=stability_by_candidate,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    variant_records_path = output_dir / "stage_c2_visibility_records.jsonl"
    output_summary_path = output_dir / "stage_c2_visibility_summary.json"
    docs_summary_path = docs_output_dir / "stage_c2_visibility_summary.json"
    docs_markdown_path = docs_output_dir / "run_log.md"

    summary = {
        **common_summary,
        "status": "complete",
        "assessment": assessment,
        "candidate_count": len(records),
        "variant_count": len(variant_records),
        "conventions": {
            "pose_convention": "ScanNet 4x4 camera-to-world; translation is camera center in world coordinates.",
            "projection_convention": "OpenCV pinhole camera coordinates; +Z is in front; u=fx*x/z+cx, v=fy*y/z+cy; pixel bounds are half-open [0,width)x[0,height).",
            "image_size_wh": list(image_size_wh),
            "intrinsics_source": str(intrinsics_path),
            "intrinsics": intrinsics.to_dict(),
            "image_size_scaling": "intrinsic.txt matches RGB size; no scaling applied.",
            "occlusion_source": "GT point-cloud z-buffer over the fixed deterministic surface sample; no RGB-D depth maps were found in posed_images/scene0000_00.",
            "depth_tolerance_unit": "meters in ScanNet GT coordinates",
        },
        "config": {
            "gt_voxel_sizes": voxel_sizes,
            "visibility_depth_tolerances": depth_tolerances,
            "surface_sample_seed_offsets": seed_offsets,
            "max_surface_points": max_surface_points,
            "max_prediction_points": max_prediction_points,
            "surface_sample_method": sample_method,
            "prediction_sample_method": prediction_sample_method,
            "coverage_thresholds": list(coverage_thresholds),
            "global_outlier_threshold": outlier_threshold,
            "novel_connectivity_radius": connectivity_radius,
            "novel_connectivity_min_largest_component_count": int(value(args, config, "novel-connectivity-min-largest-component-count", 10)),
        },
        "nominal_key": nominal_key,
        "nominal_candidate_rows": nominal_candidate_rows,
        "stability_by_candidate": stability_by_candidate,
        "ranking_stability": {
            "variant_count": len(ranking_variants),
            "strict_best_candidate_counts_by_gain_0.05": dict(sorted(best_counts.items())),
            "best_candidate_tie_counts_by_gain_0.05": dict(sorted(best_tie_counts.items())),
            "variants": ranking_variants,
        },
        "test_count": {
            "synthetic_visibility_tests": int(value(args, config, "synthetic-visibility-test-count", 7)),
        },
        "outputs": {
            "variant_records_jsonl": str(variant_records_path),
            "output_summary_json": str(output_summary_path),
            "docs_summary_json": str(docs_summary_path),
            "docs_markdown": str(docs_markdown_path),
        },
    }
    summary["runtime_seconds"] = time.perf_counter() - started_at

    write_jsonl(variant_records_path, variant_records)
    write_json(output_summary_path, summary)
    write_json(docs_summary_path, summary)
    write_markdown(docs_markdown_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "stage_c2_passed": assessment["stage_c2_passed"],
        "candidate_count": len(records),
        "variant_count": len(variant_records),
        "summary": str(docs_summary_path),
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
