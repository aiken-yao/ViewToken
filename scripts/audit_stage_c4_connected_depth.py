"""Stage-C4 connected-novel mining and v3 point-head depth diagnostics.

This script is intentionally offline-only: it uses ScanNet GT poses, calibrated
intrinsics, GT points, and existing v3 reconstruction caches. It does not run
VGGT, expand audit20, or train a policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewtoken.oracle import (  # noqa: E402
    PinholeIntrinsics,
    VisibilityMasks,
    camera_pose_delta_to_observed,
    compute_batch_preprocess_transforms,
    decode_vggt_pose_enc,
    infer_image_size_hw,
    load_point_cloud,
    load_pose_enc,
    load_pose_matrix,
    local_camera_points_to_known_world,
    predicted_world_to_local_camera_points,
    reshape_flattened_points_by_view,
    summarize_visibility_masks,
    transform_intrinsics,
    transform_world_to_camera,
    union_visible_surface_mask,
    validate_reconstruction_cache,
    visible_surface_mask,
    view_id_from_path,
)
from viewtoken.oracle.metrics import nearest_neighbor_squared_distances, sample_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audit-records", type=Path, default=None)
    parser.add_argument("--c3-summary", type=Path, default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--posed-image-dir", type=Path, default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
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


def optional_path(args: argparse.Namespace, config: dict[str, Any], name: str) -> Path | None:
    raw = value(args, config, name, None)
    if raw is None:
        return None
    return resolve_path(raw, name)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def load_intrinsics_matrix(path: Path) -> torch.Tensor:
    matrix = torch.tensor(np.loadtxt(path, dtype=np.float64), dtype=torch.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3 or 4x4, got {tuple(matrix.shape)}")
    return matrix


def load_intrinsics(path: Path, image_size_wh: tuple[int, int]) -> PinholeIntrinsics:
    matrix = load_intrinsics_matrix(path)
    return PinholeIntrinsics.from_matrix(matrix, width=image_size_wh[0], height=image_size_wh[1])


def scene_pose_ids(posed_image_dir: Path) -> list[str]:
    return sorted(
        path.stem
        for path in posed_image_dir.glob("*.txt")
        if path.stem != "intrinsic" and path.stem.isdigit()
    )


def percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.float().cpu(), q).item())


def distribution(values: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def metric_summary(values: list[float | int | None]) -> dict[str, Any]:
    finite = [float(item) for item in values if item is not None and math.isfinite(float(item))]
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


def quantile_summary(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for row in rows:
        payload: Any = row
        for key in metric_path:
            payload = payload[key]
        if payload is not None and math.isfinite(float(payload)):
            values.append(float(payload))
    return distribution(torch.tensor(values, dtype=torch.float32))


def nearest_distances(source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return nearest_neighbor_squared_distances(source, target, chunk_size=chunk_size).sqrt()


def coverage_payload(distances: torch.Tensor, thresholds: tuple[float, ...]) -> dict[str, Any]:
    distances = torch.as_tensor(distances, dtype=torch.float32).flatten().cpu()
    payload: dict[str, Any] = {"target_count": int(distances.numel()), "thresholds": {}}
    for threshold in thresholds:
        key = f"{threshold:g}"
        if distances.numel() == 0:
            payload["thresholds"][key] = {"covered_count": 0, "covered_ratio": None}
        else:
            covered = int((distances <= float(threshold)).sum().item())
            payload["thresholds"][key] = {
                "covered_count": covered,
                "covered_ratio": float(covered) / float(distances.numel()),
            }
    return payload


def build_masks_from_observed(
    observed_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> VisibilityMasks:
    overlap = observed_mask & candidate_mask
    novel = candidate_mask & ~observed_mask
    union = observed_mask | candidate_mask
    return VisibilityMasks(
        observed=observed_mask,
        candidate=candidate_mask,
        overlap=overlap,
        novel=novel,
        union=union,
    )


def compute_candidate_visibility_row(
    candidate_id: str,
    candidate_pose: torch.Tensor,
    points: torch.Tensor,
    observed_mask: torch.Tensor,
    observed_poses: dict[str, torch.Tensor],
    intrinsics: PinholeIntrinsics,
    depth_tolerance: float,
    pixel_radius: int,
    posed_image_dir: Path,
    image_extension: str,
) -> tuple[dict[str, Any], VisibilityMasks]:
    candidate_mask = visible_surface_mask(
        points,
        camera_to_world=candidate_pose,
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        pixel_radius=pixel_radius,
    )
    masks = build_masks_from_observed(observed_mask, candidate_mask)
    stats = summarize_visibility_masks(masks).to_dict()
    delta = camera_pose_delta_to_observed(candidate_pose, observed_poses).to_dict()
    row = {
        "candidate_view_id": candidate_id,
        "visibility_stats": stats,
        "connected_novel": int(stats["novel_count"]) > 0 and int(stats["overlap_count"]) > 0,
        "rgb_exists": (posed_image_dir / f"{candidate_id}.{image_extension}").is_file(),
        "pose_path": str(posed_image_dir / f"{candidate_id}.txt"),
        "camera_delta_to_observed": delta,
    }
    return row, masks


def mine_connected_candidates(
    points: torch.Tensor,
    pose_ids: list[str],
    observed_ids: list[str],
    posed_image_dir: Path,
    intrinsics: PinholeIntrinsics,
    depth_tolerance: float,
    pixel_radius: int,
    image_extension: str,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    observed_poses = {view_id: load_pose(posed_image_dir / f"{view_id}.txt") for view_id in observed_ids}
    observed_mask = union_visible_surface_mask(
        points,
        observed_poses.values(),
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        pixel_radius=pixel_radius,
    )
    rows: list[dict[str, Any]] = []
    observed_set = set(observed_ids)
    candidate_ids = [view_id for view_id in pose_ids if view_id not in observed_set]
    for index, candidate_id in enumerate(candidate_ids, start=1):
        pose = load_pose(posed_image_dir / f"{candidate_id}.txt")
        row, _masks = compute_candidate_visibility_row(
            candidate_id=candidate_id,
            candidate_pose=pose,
            points=points,
            observed_mask=observed_mask,
            observed_poses=observed_poses,
            intrinsics=intrinsics,
            depth_tolerance=depth_tolerance,
            pixel_radius=pixel_radius,
            posed_image_dir=posed_image_dir,
            image_extension=image_extension,
        )
        rows.append(row)
        if index % 100 == 0:
            print(f"mined {index}/{len(candidate_ids)} candidate poses")
    return rows, observed_mask


def stat(row: dict[str, Any], key: str) -> float:
    value = row["visibility_stats"].get(key)
    return 0.0 if value is None else float(value)


def select_connected_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    connected = [row for row in rows if row["connected_novel"] and row["rgb_exists"]]
    distributions = {
        "all_candidates": {
            "candidate_fraction": quantile_summary(rows, ("visibility_stats", "candidate_fraction")),
            "candidate_overlap_fraction": quantile_summary(rows, ("visibility_stats", "candidate_overlap_fraction")),
            "novel_scene_fraction": quantile_summary(rows, ("visibility_stats", "novel_scene_fraction")),
        },
        "connected_novel_candidates": {
            "candidate_fraction": quantile_summary(connected, ("visibility_stats", "candidate_fraction")),
            "candidate_overlap_fraction": quantile_summary(connected, ("visibility_stats", "candidate_overlap_fraction")),
            "novel_scene_fraction": quantile_summary(connected, ("visibility_stats", "novel_scene_fraction")),
        },
    }
    q = distributions["connected_novel_candidates"]
    selected: dict[str, Any] = {}
    if not connected:
        return {
            "selection_rule": {
                "pool": "non-observed poses with RGB, novel_count > 0, overlap_count > 0",
                "status": "empty_pool",
            },
            "selected_by_rule": selected,
            "selected_unique_candidate_ids": [],
            "distributions": distributions,
            "top_connected_by_novel_scene_fraction": [],
            "top_connected_by_overlap_fraction": [],
        }

    overlap_q25 = q["candidate_overlap_fraction"]["p25"]
    overlap_q50 = q["candidate_overlap_fraction"]["median"]
    overlap_q75 = q["candidate_overlap_fraction"]["p75"]
    novel_q25 = q["novel_scene_fraction"]["p25"]
    novel_q50 = q["novel_scene_fraction"]["median"]
    novel_q75 = q["novel_scene_fraction"]["p75"]
    novel_q90 = q["novel_scene_fraction"]["p90"]

    def add_selection(name: str, candidates: list[dict[str, Any]], reason: str) -> None:
        if candidates:
            row = candidates[0]
            selected[name] = {
                "candidate_view_id": row["candidate_view_id"],
                "reason": reason,
                "visibility_stats": row["visibility_stats"],
                "camera_delta_to_observed": row["camera_delta_to_observed"],
                "ranked_pool_count": len(candidates),
            }
        else:
            selected[name] = {"candidate_view_id": None, "reason": reason, "ranked_pool_count": 0}

    high_overlap = [
        row
        for row in connected
        if stat(row, "candidate_overlap_fraction") >= float(overlap_q75)
        and stat(row, "novel_scene_fraction") <= float(novel_q25)
    ]
    high_overlap.sort(key=lambda row: (-stat(row, "candidate_overlap_fraction"), stat(row, "novel_scene_fraction"), row["candidate_view_id"]))
    add_selection(
        "high_overlap_low_novel",
        high_overlap,
        "overlap >= connected p75 and novel_scene <= connected p25; sort by overlap desc, novelty asc, id asc",
    )

    medium = [
        row
        for row in connected
        if float(overlap_q25) <= stat(row, "candidate_overlap_fraction") <= float(overlap_q75)
        and float(novel_q25) <= stat(row, "novel_scene_fraction") <= float(novel_q75)
    ]
    medium.sort(
        key=lambda row: (
            abs(stat(row, "candidate_overlap_fraction") - float(overlap_q50))
            + abs(stat(row, "novel_scene_fraction") - float(novel_q50)),
            row["candidate_view_id"],
        )
    )
    add_selection(
        "medium_overlap_medium_novel",
        medium,
        "overlap and novel_scene both within connected p25-p75; sort by distance to connected medians, id asc",
    )

    high_novel = [row for row in connected if stat(row, "novel_scene_fraction") >= float(novel_q90)]
    high_novel.sort(key=lambda row: (-stat(row, "novel_scene_fraction"), -stat(row, "candidate_overlap_fraction"), row["candidate_view_id"]))
    add_selection(
        "high_novel_connected",
        high_novel,
        "novel_scene >= connected p90; sort by novelty desc, overlap desc, id asc",
    )

    nearest = sorted(
        connected,
        key=lambda row: (
            float(row["camera_delta_to_observed"]["min_distance_to_observed_meters"]),
            row["candidate_view_id"],
        ),
    )
    add_selection("nearest_connected", nearest, "minimum distance to any observed camera among connected-novel RGB poses")

    farthest = sorted(
        connected,
        key=lambda row: (
            -float(row["camera_delta_to_observed"]["min_distance_to_observed_meters"]),
            row["candidate_view_id"],
        ),
    )
    add_selection("farthest_connected", farthest, "maximum distance to any observed camera among connected-novel RGB poses")

    unique_ids: list[str] = []
    for payload in selected.values():
        candidate_id = payload.get("candidate_view_id")
        if candidate_id is not None and candidate_id not in unique_ids:
            unique_ids.append(str(candidate_id))

    return {
        "selection_rule": {
            "pool": "non-observed poses with RGB, novel_count > 0, overlap_count > 0",
            "status": "complete",
            "thresholds": {
                "connected_overlap_p25": overlap_q25,
                "connected_overlap_median": overlap_q50,
                "connected_overlap_p75": overlap_q75,
                "connected_novel_scene_p25": novel_q25,
                "connected_novel_scene_median": novel_q50,
                "connected_novel_scene_p75": novel_q75,
                "connected_novel_scene_p90": novel_q90,
            },
        },
        "selected_by_rule": selected,
        "selected_unique_candidate_ids": unique_ids,
        "distributions": distributions,
        "top_connected_by_novel_scene_fraction": [
            slim_candidate(row)
            for row in sorted(connected, key=lambda row: (-stat(row, "novel_scene_fraction"), row["candidate_view_id"]))[:20]
        ],
        "top_connected_by_overlap_fraction": [
            slim_candidate(row)
            for row in sorted(connected, key=lambda row: (-stat(row, "candidate_overlap_fraction"), row["candidate_view_id"]))[:20]
        ],
    }


def slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_view_id": row["candidate_view_id"],
        "visibility_stats": row["visibility_stats"],
        "camera_delta_to_observed": row["camera_delta_to_observed"],
        "rgb_exists": row["rgb_exists"],
        "connected_novel": row["connected_novel"],
    }


def run_stability_checks(
    target_points: torch.Tensor,
    candidate_ids: list[str],
    observed_ids: list[str],
    posed_image_dir: Path,
    intrinsics: PinholeIntrinsics,
    image_extension: str,
    point_limits: list[int],
    depth_tolerances: list[float],
    pixel_radii: list[int],
    seed: int,
    sample_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidate_ids:
        return [], {}
    pose_ids = set(scene_pose_ids(posed_image_dir))
    valid_candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id in pose_ids]
    observed_poses = {view_id: load_pose(posed_image_dir / f"{view_id}.txt") for view_id in observed_ids}
    candidate_poses = {view_id: load_pose(posed_image_dir / f"{view_id}.txt") for view_id in valid_candidate_ids}
    rows: list[dict[str, Any]] = []
    for point_limit in point_limits:
        surface_points = sample_points(
            target_points,
            max_points=int(point_limit),
            seed=seed,
            method=sample_method,
        )
        for depth_tolerance in depth_tolerances:
            for pixel_radius in pixel_radii:
                observed_mask = union_visible_surface_mask(
                    surface_points,
                    observed_poses.values(),
                    intrinsics=intrinsics,
                    depth_tolerance=depth_tolerance,
                    pixel_radius=int(pixel_radius),
                )
                for candidate_id, candidate_pose in candidate_poses.items():
                    row, _masks = compute_candidate_visibility_row(
                        candidate_id=candidate_id,
                        candidate_pose=candidate_pose,
                        points=surface_points,
                        observed_mask=observed_mask,
                        observed_poses=observed_poses,
                        intrinsics=intrinsics,
                        depth_tolerance=depth_tolerance,
                        pixel_radius=int(pixel_radius),
                        posed_image_dir=posed_image_dir,
                        image_extension=image_extension,
                    )
                    row["variant_key"] = {
                        "surface_point_limit": int(point_limit),
                        "surface_point_count": int(surface_points.shape[0]),
                        "visibility_depth_tolerance": float(depth_tolerance),
                        "visibility_pixel_radius": int(pixel_radius),
                    }
                    rows.append(row)
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate_view_id"], []).append(row)
    summary = {
        candidate_id: {
            "variant_count": len(items),
            "connected_variant_count": sum(1 for item in items if item["connected_novel"]),
            "candidate_overlap_fraction": metric_summary(
                [item["visibility_stats"]["candidate_overlap_fraction"] for item in items]
            ),
            "novel_scene_fraction": metric_summary(
                [item["visibility_stats"]["novel_scene_fraction"] for item in items]
            ),
            "overlap_count": metric_summary([item["visibility_stats"]["overlap_count"] for item in items]),
            "novel_count": metric_summary([item["visibility_stats"]["novel_count"] for item in items]),
        }
        for candidate_id, items in sorted(by_candidate.items())
    }
    return rows, summary


def expected_reconstruction_fingerprint(record: dict[str, Any], role: str) -> str | None:
    metadata = record.get("metadata", {})
    reconstruction = metadata.get(f"{role}_reconstruction", {})
    fingerprint = reconstruction.get("cache_fingerprint")
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def reconstruction_dir(record: dict[str, Any], role: str) -> Path:
    return Path(record["reconstruction_paths"][role]).expanduser().resolve().parent


def cache_metadata(record: dict[str, Any], role: str) -> dict[str, Any]:
    return validate_reconstruction_cache(
        reconstruction_dir(record, role),
        expected_fingerprint=expected_reconstruction_fingerprint(record, role),
    )


def load_cache_image_view_ids(metadata: dict[str, Any]) -> list[str]:
    return [view_id_from_path(Path(path)) for path in metadata["image_paths"]]


def alignment_scale(record: dict[str, Any], role: str) -> float:
    return float(record["metadata"][f"{role}_camera_anchor_alignment"]["transform"]["scale"])


def load_candidate_view_geometry(
    record: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    metadata = cache_metadata(record, "candidate")
    cache_dir = reconstruction_dir(record, "candidate")
    points = load_point_cloud(cache_dir / "points.pt")
    per_view_world, flatten = reshape_flattened_points_by_view(points, metadata)
    confidence = torch.load(cache_dir / "confidence.pt", map_location="cpu", weights_only=True).float()
    expected = int(flatten.expected_point_count)
    if int(confidence.numel()) != expected:
        raise RuntimeError(f"confidence.pt has {confidence.numel()} values, expected {expected}")
    confidence = confidence.reshape(flatten.view_count, flatten.height, flatten.width)
    pose_enc = load_pose_enc(cache_dir / "pose_enc.pt")
    extrinsics, predicted_intrinsics = decode_vggt_pose_enc(
        pose_enc,
        image_size_hw=infer_image_size_hw(metadata),
        build_intrinsics=True,
    )
    extrinsics = extrinsics.squeeze(0).float().cpu()
    predicted_intrinsics = predicted_intrinsics.squeeze(0).float().cpu()
    local = predicted_world_to_local_camera_points(per_view_world, extrinsics)
    image_view_ids = load_cache_image_view_ids(metadata)
    if candidate_id not in image_view_ids:
        raise RuntimeError(f"candidate {candidate_id} not present in cache image order {image_view_ids}")
    candidate_index = image_view_ids.index(candidate_id)
    return {
        "metadata": metadata,
        "cache_dir": str(cache_dir),
        "image_view_ids": image_view_ids,
        "candidate_index": candidate_index,
        "per_view_world_points": per_view_world,
        "per_view_local_points": local,
        "candidate_local_points": local[candidate_index],
        "candidate_confidence": confidence[candidate_index],
        "predicted_intrinsics": predicted_intrinsics[candidate_index],
        "all_predicted_intrinsics": predicted_intrinsics,
        "flatten": flatten.to_dict(),
        "depth_scale_from_candidate_camera_anchor_sim3": alignment_scale(record, "candidate"),
    }


def transformed_calibrated_intrinsics_for_cache(
    metadata: dict[str, Any],
    intrinsics_matrix: torch.Tensor,
    preprocess_mode: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    sizes = [read_image_size(Path(path)) for path in metadata["image_paths"]]
    transforms = compute_batch_preprocess_transforms(
        sizes,
        mode=preprocess_mode,
        target_size=518,
        patch_size=14,
    )
    transformed = torch.stack(
        [transform_intrinsics(intrinsics_matrix, transform) for transform in transforms],
        dim=0,
    )
    return transformed, [transform.to_dict() for transform in transforms]


def coordinate_distributions(points: torch.Tensor) -> dict[str, Any]:
    flat = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3).cpu()
    finite = torch.isfinite(flat).all(dim=-1)
    flat = flat[finite]
    if flat.numel() == 0:
        radial = torch.empty((0,), dtype=torch.float32)
        xy_radial = torch.empty((0,), dtype=torch.float32)
    else:
        radial = torch.linalg.norm(flat, dim=-1)
        xy_radial = torch.linalg.norm(flat[:, :2], dim=-1)
    return {
        "finite_count": int(flat.shape[0]),
        "x": distribution(flat[:, 0] if flat.numel() else torch.empty((0,))),
        "y": distribution(flat[:, 1] if flat.numel() else torch.empty((0,))),
        "z_depth": distribution(flat[:, 2] if flat.numel() else torch.empty((0,))),
        "xy_radius": distribution(xy_radial),
        "radial_depth": distribution(radial),
        "positive_z_ratio": None if flat.numel() == 0 else float((flat[:, 2] > 0.0).float().mean().item()),
    }


def confidence_quantiles(confidence: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, Any]:
    flat = torch.as_tensor(confidence, dtype=torch.float32).flatten().cpu()
    finite = flat[torch.isfinite(flat)]
    payload = distribution(finite)
    payload["quantile_thresholds"] = {
        f"q{quantile:g}": None if finite.numel() == 0 else float(torch.quantile(finite, quantile).item())
        for quantile in quantiles
    }
    return payload


def ray_consistency(
    local_points: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> dict[str, Any]:
    local = torch.as_tensor(local_points, dtype=torch.float32).reshape(-1, 3).cpu()
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32).cpu()
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    finite = torch.isfinite(local).all(dim=-1) & (local[:, 2] > 1e-6)
    if not bool(finite.any()):
        return {
            "valid_count": 0,
            "pixel_reprojection_error": distribution(torch.empty((0,))),
            "ray_angle_degrees": distribution(torch.empty((0,))),
            "projected_in_frame_ratio": None,
        }
    local_valid = local[finite]
    pixels_valid = pixels[finite]
    fx = intrinsics[0, 0].clamp_min(1e-6)
    fy = intrinsics[1, 1].clamp_min(1e-6)
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    projected = torch.empty_like(pixels_valid)
    projected[:, 0] = fx * local_valid[:, 0] / local_valid[:, 2] + cx
    projected[:, 1] = fy * local_valid[:, 1] / local_valid[:, 2] + cy
    reprojection_error = torch.linalg.norm(projected - pixels_valid, dim=-1)
    ideal_rays = torch.stack(
        [
            (pixels_valid[:, 0] - cx) / fx,
            (pixels_valid[:, 1] - cy) / fy,
            torch.ones_like(pixels_valid[:, 0]),
        ],
        dim=-1,
    )
    local_rays = local_valid / torch.linalg.norm(local_valid, dim=-1, keepdim=True).clamp_min(1e-12)
    ideal_rays = ideal_rays / torch.linalg.norm(ideal_rays, dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = (local_rays * ideal_rays).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    in_frame = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] < float(width))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < float(height))
    )
    return {
        "valid_count": int(local_valid.shape[0]),
        "pixel_reprojection_error": distribution(reprojection_error),
        "ray_angle_degrees": distribution(angles),
        "projected_in_frame_ratio": float(in_frame.float().mean().item()),
    }


def confidence_sweep_diagnostics(
    candidate_world_points: torch.Tensor,
    candidate_confidence: torch.Tensor,
    gt_target_points: torch.Tensor,
    gt_novel_points: torch.Tensor,
    quantiles: tuple[float, ...],
    coverage_thresholds: tuple[float, ...],
    outlier_threshold: float,
    chunk_size: int,
    max_outlier_points: int,
    seed: int,
) -> list[dict[str, Any]]:
    world = torch.as_tensor(candidate_world_points, dtype=torch.float32).reshape(-1, 3).cpu()
    confidence = torch.as_tensor(candidate_confidence, dtype=torch.float32).reshape(-1).cpu()
    finite = torch.isfinite(world).all(dim=-1) & torch.isfinite(confidence)
    world = world[finite]
    confidence = confidence[finite]
    rows = []
    for quantile in quantiles:
        threshold = float(torch.quantile(confidence, float(quantile)).item()) if confidence.numel() else math.inf
        keep = confidence >= threshold
        filtered = world[keep]
        if filtered.numel() == 0:
            rows.append(
                {
                    "confidence_quantile": float(quantile),
                    "confidence_threshold": threshold,
                    "kept_point_count": 0,
                    "kept_point_ratio": 0.0,
                    "novel_coverage": coverage_payload(torch.empty((0,)), coverage_thresholds),
                    "novel_distance_distribution": distribution(torch.empty((0,))),
                    "outlier_ratio_to_full_gt": None,
                    "outlier_threshold_meters": outlier_threshold,
                }
            )
            continue
        if gt_novel_points.numel():
            novel_distances = nearest_distances(gt_novel_points, filtered, chunk_size=chunk_size)
        else:
            novel_distances = torch.empty((0,), dtype=torch.float32)
        sampled_filtered = sample_points(
            filtered,
            max_points=max_outlier_points,
            seed=seed + int(round(quantile * 1000.0)),
            method="hash",
        )
        pred_to_gt = nearest_distances(sampled_filtered, gt_target_points, chunk_size=chunk_size)
        rows.append(
            {
                "confidence_quantile": float(quantile),
                "confidence_threshold": threshold,
                "kept_point_count": int(filtered.shape[0]),
                "kept_point_ratio": float(filtered.shape[0]) / float(max(1, world.shape[0])),
                "novel_coverage": coverage_payload(novel_distances, coverage_thresholds),
                "novel_distance_distribution": distribution(novel_distances),
                "outlier_ratio_to_full_gt": float((pred_to_gt > float(outlier_threshold)).float().mean().item()),
                "outlier_threshold_meters": outlier_threshold,
                "outlier_sample_count": int(sampled_filtered.shape[0]),
            }
        )
    return rows


def extract_c3_branch_comparison(c3_summary_path: Path | None, candidate_id: str) -> dict[str, Any]:
    if c3_summary_path is None or not c3_summary_path.is_file():
        return {"status": "missing_c3_summary"}
    summary = load_json(c3_summary_path)
    for row in summary.get("candidate_rows", []):
        if str(row.get("candidate_view_id")) == candidate_id:
            branches = row.get("branches", {})
            return {
                "status": "loaded_from_stage_c3",
                "predicted_world": branches.get("predicted_world"),
                "point_head_known_pose": branches.get("known_pose"),
                "corrected_visibility_semantic_tag": row.get("corrected_visibility_semantic_tag"),
                "stage_c3_passed": summary.get("assessment", {}).get("stage_c3_passed"),
            }
    return {"status": "candidate_missing_in_c3_summary", "candidate_view_id": candidate_id}


def diagnose_v3_candidate(
    record: dict[str, Any],
    candidate_id: str,
    target_points: torch.Tensor,
    gt_novel_points: torch.Tensor,
    candidate_pose: torch.Tensor,
    intrinsics_matrix: torch.Tensor,
    preprocess_mode: str,
    confidence_quantile_values: tuple[float, ...],
    coverage_thresholds: tuple[float, ...],
    outlier_threshold: float,
    chunk_size: int,
    max_outlier_points: int,
    seed: int,
    c3_summary_path: Path | None,
) -> dict[str, Any]:
    geometry = load_candidate_view_geometry(record, candidate_id)
    candidate_local = geometry["candidate_local_points"]
    candidate_confidence = geometry["candidate_confidence"]
    height = int(candidate_local.shape[0])
    width = int(candidate_local.shape[1])
    scale = float(geometry["depth_scale_from_candidate_camera_anchor_sim3"])
    known_world = local_camera_points_to_known_world(
        candidate_local.unsqueeze(0),
        candidate_pose.unsqueeze(0),
        depth_scale=scale,
    )[0].reshape(-1, 3)
    transformed_gt_intrinsics, preprocessing_transforms = transformed_calibrated_intrinsics_for_cache(
        geometry["metadata"],
        intrinsics_matrix,
        preprocess_mode=preprocess_mode,
    )
    candidate_index = int(geometry["candidate_index"])
    gt_local_novel = transform_world_to_camera(gt_novel_points, candidate_pose) if gt_novel_points.numel() else torch.empty((0, 3))
    return {
        "candidate_view_id": candidate_id,
        "cache_dir": geometry["cache_dir"],
        "cache_image_view_ids": geometry["image_view_ids"],
        "candidate_index": candidate_index,
        "flatten": geometry["flatten"],
        "depth_scale_from_candidate_camera_anchor_sim3": scale,
        "world_points_confidence": confidence_quantiles(candidate_confidence, confidence_quantile_values),
        "local_camera_space_raw_point_head": coordinate_distributions(candidate_local),
        "local_camera_space_scaled_by_anchor_sim3": coordinate_distributions(candidate_local * scale),
        "gt_novel_points_in_candidate_camera_frame": coordinate_distributions(gt_local_novel),
        "ray_consistency": {
            "point_head_local_vs_predicted_intrinsics": ray_consistency(
                candidate_local,
                geometry["predicted_intrinsics"],
                width=width,
                height=height,
            ),
            "point_head_local_vs_transformed_calibrated_intrinsics": ray_consistency(
                candidate_local,
                transformed_gt_intrinsics[candidate_index],
                width=width,
                height=height,
            ),
            "predicted_intrinsics": geometry["predicted_intrinsics"].tolist(),
            "transformed_calibrated_intrinsics": transformed_gt_intrinsics[candidate_index].tolist(),
            "preprocessing_transform": preprocessing_transforms[candidate_index],
        },
        "confidence_quantile_sweep": confidence_sweep_diagnostics(
            candidate_world_points=known_world,
            candidate_confidence=candidate_confidence,
            gt_target_points=target_points,
            gt_novel_points=gt_novel_points,
            quantiles=confidence_quantile_values,
            coverage_thresholds=coverage_thresholds,
            outlier_threshold=outlier_threshold,
            chunk_size=chunk_size,
            max_outlier_points=max_outlier_points,
            seed=seed,
        ),
        "stage_c3_branch_comparison": extract_c3_branch_comparison(c3_summary_path, candidate_id),
        "v4_depth_branches": {
            "C_depth_predicted_intrinsics_known_pose": {
                "status": "implemented_not_run_no_v4_cache",
                "required_artifacts": ["depth.pt", "depth_conf.pt", "predicted_intrinsics.pt", "pose_enc.pt"],
            },
            "D_depth_calibrated_intrinsics_known_pose": {
                "status": "implemented_not_run_no_v4_cache",
                "required_artifacts": ["depth.pt", "depth_conf.pt", "transformed_gt_intrinsics.pt", "preprocessing_transforms", "pose_enc.pt"],
            },
        },
    }


def build_diagnostic_assessment(diagnostic: dict[str, Any]) -> dict[str, Any]:
    visibility = diagnostic.get("nominal_gt_visibility_stats", {})
    conf = diagnostic.get("world_points_confidence", {})
    local_z = diagnostic.get("local_camera_space_raw_point_head", {}).get("z_depth", {})
    gt_z = diagnostic.get("gt_novel_points_in_candidate_camera_frame", {}).get("z_depth", {})
    ray = diagnostic.get("ray_consistency", {})
    pred_ray = ray.get("point_head_local_vs_predicted_intrinsics", {})
    calib_ray = ray.get("point_head_local_vs_transformed_calibrated_intrinsics", {})
    sweep = diagnostic.get("confidence_quantile_sweep", [])
    best_covered_005 = 0
    best_covered_010 = 0
    for row in sweep:
        thresholds = row.get("novel_coverage", {}).get("thresholds", {})
        best_covered_005 = max(best_covered_005, int(thresholds.get("0.05", {}).get("covered_count") or 0))
        best_covered_010 = max(best_covered_010, int(thresholds.get("0.1", {}).get("covered_count") or 0))
    conf_range = None
    if conf.get("min") is not None and conf.get("max") is not None:
        conf_range = float(conf["max"]) - float(conf["min"])
    local_median = local_z.get("median")
    gt_median = gt_z.get("median")
    local_behind = local_median is not None and float(local_median) <= 0.0
    gt_in_front = gt_median is not None and float(gt_median) > 0.0
    ray_valid_pred = int(pred_ray.get("valid_count") or 0)
    ray_valid_calib = int(calib_ray.get("valid_count") or 0)
    likely_failure = "undetermined"
    evidence = []
    if int(visibility.get("overlap_count") or 0) == 0 and int(visibility.get("novel_count") or 0) > 0:
        evidence.append("GT visibility marks 00425 as disconnected novel: novel_count > 0 and overlap_count == 0")
    if local_behind and gt_in_front:
        likely_failure = "point_head_local_geometry_behind_candidate_camera"
        evidence.append("point-head local Z median is <= 0 while GT novel Z median is > 0")
    if conf_range is not None and conf_range < 1e-3:
        evidence.append("candidate-view confidence is nearly constant, so quantile filtering cannot separate good geometry")
    if best_covered_005 == 0 and best_covered_010 == 0:
        evidence.append("confidence sweep recovered zero GT novel points at 0.05m and 0.10m")
    if ray_valid_pred == 0 and ray_valid_calib == 0:
        evidence.append("ray/intrinsics checks have zero positive-Z local points to project")
    return {
        "likely_primary_failure": likely_failure,
        "is_low_confidence_failure": False if conf_range is not None and conf_range < 1e-3 else None,
        "confidence_range": conf_range,
        "best_novel_covered_count_0.05": best_covered_005,
        "best_novel_covered_count_0.10": best_covered_010,
        "ray_consistency_has_positive_z_points": bool(ray_valid_pred or ray_valid_calib),
        "evidence": evidence,
    }


def make_proposed_v4_config(
    selected_ids: list[str],
    observed_ids: list[str],
    posed_image_dir: Path,
    target_points_path: Path,
    checkpoint_path: Path | None,
    intrinsics_path: Path,
    docs_output_dir: Path,
    image_extension: str,
) -> Path:
    config_path = docs_output_dir / "proposed_v4_connected_candidates.yaml"
    payload = {
        "scene-id": "scene0000_00",
        "observed-views": [str(posed_image_dir / f"{view_id}.{image_extension}") for view_id in observed_ids],
        "candidate-views": [str(posed_image_dir / f"{view_id}.{image_extension}") for view_id in selected_ids],
        "target-points": str(target_points_path),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "output-dir": "outputs/oracle_gain/scannet_scene0000_00_stage_c4_v4_connected",
        "preprocess-mode": "crop",
        "cache-schema-version": "v4",
        "calibrated-intrinsics": str(intrinsics_path),
        "min-world-point-confidence": 0.0,
        "max-reconstruction-points": None,
        "reconstruction-sample-method": "none",
        "max-metric-points": 12000,
        "metric-sample-method": "hash",
        "alignment": "sim3_icp",
        "random-seed": 0,
        "point-stride": 6,
        "reuse-reconstructions": False,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path


def markdown_table(rows: list[dict[str, Any]], limit: int = 12) -> str:
    lines = []
    for row in rows[:limit]:
        stats = row["visibility_stats"]
        delta = row["camera_delta_to_observed"]
        lines.append(
            "| `{candidate}` | `{overlap}` | `{novel}` | `{visible}` | `{distance}` | `{angle}` |".format(
                candidate=row["candidate_view_id"],
                overlap=f"{(stats['candidate_overlap_fraction'] or 0.0):.6f}",
                novel=f"{stats['novel_scene_fraction']:.6f}",
                visible=f"{stats['candidate_fraction']:.6f}",
                distance=f"{delta['min_distance_to_observed_meters']:.4f}",
                angle=f"{delta['min_view_direction_change_degrees']:.2f}",
            )
        )
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    selection = summary["candidate_selection"]
    selected_rows = [
        payload
        for payload in selection["selected_by_rule"].values()
        if payload.get("candidate_view_id") is not None
    ]
    selected_lines = []
    for category, payload in selection["selected_by_rule"].items():
        selected_lines.append(
            f"- `{category}`: `{payload.get('candidate_view_id')}` ({payload.get('reason')})"
        )
    diagnostic = summary["diagnostic_00425"]
    conf = diagnostic["world_points_confidence"]
    ray_pred = diagnostic["ray_consistency"]["point_head_local_vs_predicted_intrinsics"]["pixel_reprojection_error"]
    ray_calib = diagnostic["ray_consistency"]["point_head_local_vs_transformed_calibrated_intrinsics"]["pixel_reprojection_error"]
    local_z = diagnostic["local_camera_space_raw_point_head"]["z_depth"]
    gt_z = diagnostic["gt_novel_points_in_candidate_camera_frame"]["z_depth"]
    assessment = diagnostic["assessment"]
    assessment_lines = "\n".join(f"- {item}" for item in assessment["evidence"])
    sweep_lines = []
    for row in diagnostic["confidence_quantile_sweep"]:
        cov = row["novel_coverage"]["thresholds"]
        sweep_lines.append(
            "| `{q}` | `{thr:.6f}` | `{kept}` | `{cov05}` | `{cov10}` | `{out}` |".format(
                q=row["confidence_quantile"],
                thr=row["confidence_threshold"],
                kept=row["kept_point_count"],
                cov05=cov.get("0.05", {}).get("covered_count"),
                cov10=cov.get("0.1", {}).get("covered_count"),
                out="None" if row["outlier_ratio_to_full_gt"] is None else f"{row['outlier_ratio_to_full_gt']:.6f}",
            )
        )
    text = f"""# Stage C4 Connected-Depth Diagnostics

Status: `{summary['status']}`

This run did not execute VGGT, did not expand audit20, and did not train a policy.

## Candidate Mining

- Scene pose candidates scanned: `{summary['candidate_mining']['candidate_pose_count']}`
- Connected-novel RGB candidates found: `{summary['candidate_mining']['connected_novel_rgb_count']}`
- Connected-novel definition: `novel_count > 0 && overlap_count > 0`
- Visibility source: `{summary['candidate_mining']['visibility_source']}`
- Nominal visibility: depth tolerance `{summary['config']['visibility_depth_tolerance']}`, pixel radius `{summary['config']['visibility_pixel_radius']}`

## Fixed Selection

{chr(10).join(selected_lines)}

Selected unique candidates: `{selection['selected_unique_candidate_ids']}`
Expected v4 cache count after approval: `{summary['proposed_v4_run']['expected_cache_count']}`

## Top Connected By Novelty

| cand | cand overlap frac | novel scene frac | visible scene frac | min dist m | min angle deg |
|---|---:|---:|---:|---:|---:|
{markdown_table(selection['top_connected_by_novel_scene_fraction'])}

## Top Connected By Overlap

| cand | cand overlap frac | novel scene frac | visible scene frac | min dist m | min angle deg |
|---|---:|---:|---:|---:|---:|
{markdown_table(selection['top_connected_by_overlap_fraction'])}

## Splatting Stability

Stability variants: `{summary['stability']['variant_count']}`

JSON summary contains per-candidate overlap/novel distributions for point limits `{summary['config']['stability_point_limits']}`, depth tolerances `{summary['config']['stability_depth_tolerances']}`, and pixel radii `{summary['config']['stability_pixel_radii']}`.

## 00425 v3 Diagnosis

- Candidate-view confidence median/min/max: `{conf['median']}` / `{conf['min']}` / `{conf['max']}`
- Point-head local Z median/p10/p90: `{local_z['median']}` / `{local_z['p10']}` / `{local_z['p90']}`
- GT novel local Z median/p10/p90: `{gt_z['median']}` / `{gt_z['p10']}` / `{gt_z['p90']}`
- Ray pixel error median with predicted intrinsics: `{ray_pred['median']}`
- Ray pixel error median with transformed calibrated intrinsics: `{ray_calib['median']}`
- Likely primary failure: `{assessment['likely_primary_failure']}`
- Low-confidence failure: `{assessment['is_low_confidence_failure']}`

{assessment_lines}

| conf quantile | threshold | kept points | novel covered @0.05 | novel covered @0.10 | outlier ratio |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(sweep_lines)}

## Branch Status

- A predicted-world: loaded from Stage C3 existing v3 cache when available.
- B point-head known-pose: loaded from Stage C3 and diagnosed here from candidate-view local geometry.
- C depth + predicted intrinsics + known pose: implemented in `viewtoken/oracle/depth_branch.py`, not run because no v4 cache exists yet.
- D depth + transformed calibrated intrinsics + known pose: implemented in `viewtoken/oracle/depth_branch.py`, not run because no v4 cache exists yet.

## Proposed Commands

```bash
{summary['proposed_v4_run']['exact_generate_command']}
```

```bash
{summary['proposed_v4_run']['exact_c4_command_after_v4']}
```

Full JSON summary: `{summary['outputs']['docs_summary_json']}`
Candidate records: `{summary['outputs']['candidate_records_jsonl']}`
Candidate records for GitHub: `{summary['outputs']['docs_candidate_records_jsonl']}`
Stability records: `{summary['outputs']['stability_records_jsonl']}`
Stability records for GitHub: `{summary['outputs']['docs_stability_records_jsonl']}`
Proposed v4 config: `{summary['outputs']['proposed_v4_config']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)

    audit_records_path = resolve_path(value(args, config, "audit-records", None), "audit-records")
    c3_summary_path = optional_path(args, config, "c3-summary")
    target_points_path = resolve_path(value(args, config, "target-points", None), "target-points")
    posed_image_dir = resolve_path(value(args, config, "posed-image-dir", None), "posed-image-dir")
    intrinsics_path = resolve_path(value(args, config, "intrinsics", None), "intrinsics")
    checkpoint_path = optional_path(args, config, "checkpoint")
    output_dir = resolve_path(
        value(args, config, "output-dir", "outputs/oracle_calibration/scannet_scene0000_00_stage_c4"),
        "output-dir",
        must_exist=False,
    )
    docs_output_dir = resolve_path(
        value(args, config, "docs-output-dir", "docs/audits/scannet_scene0000_00_stage_c4"),
        "docs-output-dir",
        must_exist=False,
    )

    image_extension = str(value(args, config, "image-extension", "jpg")).lstrip(".")
    point_stride = int(value(args, config, "point-stride", 6))
    depth_tolerance = float(value(args, config, "visibility-depth-tolerance", 0.05))
    pixel_radius = int(value(args, config, "visibility-pixel-radius", 0))
    stability_point_limits = [int(item) for item in value(args, config, "stability-point-limits", [12000, 50000])]
    stability_depth_tolerances = [float(item) for item in value(args, config, "stability-depth-tolerances", [0.02, 0.05, 0.10])]
    stability_pixel_radii = [int(item) for item in value(args, config, "stability-pixel-radii", [0, 1])]
    confidence_quantile_values = tuple(float(item) for item in value(args, config, "confidence-quantiles", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
    coverage_thresholds = tuple(float(item) for item in value(args, config, "coverage-thresholds", [0.05, 0.10, 0.20, 0.50]))
    outlier_threshold = float(value(args, config, "outlier-threshold", 0.10))
    chunk_size = int(value(args, config, "nn-chunk-size", 128))
    max_outlier_points = int(value(args, config, "max-outlier-points", 30000))
    seed = int(value(args, config, "seed", 0))
    sample_method = str(value(args, config, "surface-sample-method", "hash"))
    diagnostic_candidate_id = str(value(args, config, "diagnostic-candidate-id", "00425"))
    preprocess_mode = str(value(args, config, "preprocess-mode", "crop"))

    audit_records = load_jsonl(audit_records_path)
    if not audit_records:
        raise RuntimeError("audit-records is empty")
    observed_ids = [str(view_id) for view_id in audit_records[0]["observed_view_ids"]]
    target_points = load_point_cloud(target_points_path, point_stride=point_stride).float().cpu()
    pose_ids = scene_pose_ids(posed_image_dir)
    image_size_wh = read_image_size(posed_image_dir / f"{observed_ids[0]}.{image_extension}")
    intrinsics = load_intrinsics(intrinsics_path, image_size_wh=image_size_wh)
    intrinsics_matrix = load_intrinsics_matrix(intrinsics_path)

    rows, observed_mask = mine_connected_candidates(
        points=target_points,
        pose_ids=pose_ids,
        observed_ids=observed_ids,
        posed_image_dir=posed_image_dir,
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        pixel_radius=pixel_radius,
        image_extension=image_extension,
    )
    selection = select_connected_candidates(rows)
    stability_ids = list(selection["selected_unique_candidate_ids"])
    for extra_id in ("00325", diagnostic_candidate_id):
        if extra_id not in stability_ids:
            stability_ids.append(extra_id)
    stability_rows, stability_summary = run_stability_checks(
        target_points=target_points,
        candidate_ids=stability_ids,
        observed_ids=observed_ids,
        posed_image_dir=posed_image_dir,
        intrinsics=intrinsics,
        image_extension=image_extension,
        point_limits=stability_point_limits,
        depth_tolerances=stability_depth_tolerances,
        pixel_radii=stability_pixel_radii,
        seed=seed,
        sample_method=sample_method,
    )

    observed_poses = {view_id: load_pose(posed_image_dir / f"{view_id}.txt") for view_id in observed_ids}
    diagnostic_pose = load_pose(posed_image_dir / f"{diagnostic_candidate_id}.txt")
    _diag_row, diagnostic_masks = compute_candidate_visibility_row(
        candidate_id=diagnostic_candidate_id,
        candidate_pose=diagnostic_pose,
        points=target_points,
        observed_mask=observed_mask,
        observed_poses=observed_poses,
        intrinsics=intrinsics,
        depth_tolerance=depth_tolerance,
        pixel_radius=pixel_radius,
        posed_image_dir=posed_image_dir,
        image_extension=image_extension,
    )
    gt_novel_points = target_points[diagnostic_masks.novel]
    diagnostic_record = next(
        (record for record in audit_records if str(record["candidate_view_id"]) == diagnostic_candidate_id),
        None,
    )
    if diagnostic_record is None:
        raise RuntimeError(f"diagnostic candidate {diagnostic_candidate_id} is missing from v3 audit records")
    diagnostic = diagnose_v3_candidate(
        record=diagnostic_record,
        candidate_id=diagnostic_candidate_id,
        target_points=target_points,
        gt_novel_points=gt_novel_points,
        candidate_pose=diagnostic_pose,
        intrinsics_matrix=intrinsics_matrix,
        preprocess_mode=preprocess_mode,
        confidence_quantile_values=confidence_quantile_values,
        coverage_thresholds=coverage_thresholds,
        outlier_threshold=outlier_threshold,
        chunk_size=chunk_size,
        max_outlier_points=max_outlier_points,
        seed=seed,
        c3_summary_path=c3_summary_path,
    )
    diagnostic["nominal_gt_visibility_stats"] = summarize_visibility_masks(diagnostic_masks).to_dict()
    diagnostic["assessment"] = build_diagnostic_assessment(diagnostic)

    connected_count = sum(1 for row in rows if row["connected_novel"])
    connected_rgb_count = sum(1 for row in rows if row["connected_novel"] and row["rgb_exists"])
    semantic_counts = Counter(
        "connected_novel" if row["connected_novel"] else "not_connected_novel"
        for row in rows
    )
    proposed_v4_config = make_proposed_v4_config(
        selected_ids=selection["selected_unique_candidate_ids"],
        observed_ids=observed_ids,
        posed_image_dir=posed_image_dir,
        target_points_path=target_points_path,
        checkpoint_path=checkpoint_path,
        intrinsics_path=intrinsics_path,
        docs_output_dir=docs_output_dir,
        image_extension=image_extension,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_records_path = output_dir / "stage_c4_candidate_visibility_records.jsonl"
    stability_records_path = output_dir / "stage_c4_splatting_stability_records.jsonl"
    docs_candidate_records_path = docs_output_dir / "stage_c4_candidate_visibility_records.jsonl"
    docs_stability_records_path = docs_output_dir / "stage_c4_splatting_stability_records.jsonl"
    output_summary_path = output_dir / "stage_c4_connected_depth_summary.json"
    docs_summary_path = docs_output_dir / "stage_c4_connected_depth_summary.json"
    markdown_path = docs_output_dir / "run_log.md"

    python_path = "/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python"
    summary = {
        "status": "complete",
        "did_run_vggt": False,
        "did_expand_audit20": False,
        "did_train_policy": False,
        "inputs": {
            "audit_records": str(audit_records_path),
            "c3_summary": None if c3_summary_path is None else str(c3_summary_path),
            "target_points": str(target_points_path),
            "posed_image_dir": str(posed_image_dir),
            "intrinsics": str(intrinsics_path),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        },
        "config": {
            "observed_view_ids": observed_ids,
            "image_size_wh": list(image_size_wh),
            "point_stride": point_stride,
            "visibility_depth_tolerance": depth_tolerance,
            "visibility_pixel_radius": pixel_radius,
            "stability_point_limits": stability_point_limits,
            "stability_depth_tolerances": stability_depth_tolerances,
            "stability_pixel_radii": stability_pixel_radii,
            "confidence_quantiles": list(confidence_quantile_values),
            "coverage_thresholds": list(coverage_thresholds),
            "outlier_threshold": outlier_threshold,
            "nn_chunk_size": chunk_size,
            "max_outlier_points": max_outlier_points,
            "surface_sample_method": sample_method,
            "preprocess_mode": preprocess_mode,
        },
        "candidate_mining": {
            "scene_pose_count": len(pose_ids),
            "candidate_pose_count": len(rows),
            "target_point_count": int(target_points.shape[0]),
            "observed_visible_count": int(observed_mask.sum().item()),
            "connected_novel_count": connected_count,
            "connected_novel_rgb_count": connected_rgb_count,
            "semantic_counts": dict(sorted(semantic_counts.items())),
            "visibility_source": (
                "ScanNet GT point z-buffer over full 50k target points; no dense depth/mesh visibility is used. "
                "C4 adds pixel_radius splatting stability checks to expose sparse-point sensitivity."
            ),
        },
        "candidate_selection": selection,
        "stability": {
            "candidate_ids": stability_ids,
            "variant_count": len(stability_rows),
            "summary_by_candidate": stability_summary,
        },
        "diagnostic_00425": diagnostic,
        "v4_depth_branch_implementation": {
            "source_files": [
                "viewtoken/oracle/depth_branch.py",
                "viewtoken/oracle/cache.py",
                "scripts/generate_oracle_gain.py",
            ],
            "artifact_schema": [
                "depth.pt",
                "depth_conf.pt",
                "predicted_intrinsics.pt",
                "transformed_gt_intrinsics.pt",
                "preprocessing_transforms",
                "per_view_shape_offsets",
            ],
            "status": "implemented_not_run",
        },
        "proposed_v4_run": {
            "selected_candidate_ids": selection["selected_unique_candidate_ids"],
            "expected_cache_count": 1 + len(selection["selected_unique_candidate_ids"]),
            "exact_generate_command": f"{python_path} scripts/generate_oracle_gain.py --config {proposed_v4_config}",
            "exact_c4_command_after_v4": f"{python_path} scripts/audit_stage_c4_connected_depth.py --config configs/oracle_stage_c4_connected_depth.yaml",
            "requires_user_approval_before_running_vggt": True,
        },
        "outputs": {
            "candidate_records_jsonl": str(candidate_records_path),
            "stability_records_jsonl": str(stability_records_path),
            "docs_candidate_records_jsonl": str(docs_candidate_records_path),
            "docs_stability_records_jsonl": str(docs_stability_records_path),
            "output_summary_json": str(output_summary_path),
            "docs_summary_json": str(docs_summary_path),
            "docs_markdown": str(markdown_path),
            "proposed_v4_config": str(proposed_v4_config),
        },
        "validation": {
            "status": "pending_external_test_commands",
            "tests_recorded_after_script": False,
        },
        "runtime_seconds": time.perf_counter() - started_at,
    }

    sorted_candidate_rows = sorted(rows, key=lambda row: row["candidate_view_id"])
    sorted_stability_rows = sorted(stability_rows, key=lambda row: (row["candidate_view_id"], json.dumps(row["variant_key"], sort_keys=True)))
    write_jsonl(candidate_records_path, sorted_candidate_rows)
    write_jsonl(stability_records_path, sorted_stability_rows)
    write_jsonl(docs_candidate_records_path, sorted_candidate_rows)
    write_jsonl(docs_stability_records_path, sorted_stability_rows)
    write_json(output_summary_path, summary)
    write_json(docs_summary_path, summary)
    write_markdown(markdown_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "candidate_pose_count": summary["candidate_mining"]["candidate_pose_count"],
        "connected_novel_rgb_count": summary["candidate_mining"]["connected_novel_rgb_count"],
        "selected_unique_candidate_ids": selection["selected_unique_candidate_ids"],
        "did_run_vggt": summary["did_run_vggt"],
        "summary": str(docs_summary_path),
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
