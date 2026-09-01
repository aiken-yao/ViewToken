"""Generate offline oracle reconstruction-gain labels for candidate views."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt"
for import_root in (PROJECT_ROOT, VGGT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.extract_vggt_features import (  # noqa: E402
    directory_size_bytes,
    load_config,
    load_vggt_model,
    resolve_compute_dtype,
    resolve_device,
)
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from viewtoken.backbones import VGGTFeatureExtractor  # noqa: E402
from viewtoken.oracle import (  # noqa: E402
    OracleGainRecord,
    align_sim3_icp,
    build_memory_id,
    build_reconstruction_cache_identity,
    compute_pointcloud_metrics,
    load_point_cloud,
    load_pose_matrix,
    scene_split,
    summarize_oracle_audit,
    view_id_from_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--observed-views", type=Path, nargs="+", default=None)
    parser.add_argument("--candidate-views", type=Path, nargs="+", default=None)
    parser.add_argument("--target-points", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preprocess-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compute-dtype", type=str, default=None)
    parser.add_argument("--min-world-point-confidence", type=float, default=None)
    parser.add_argument("--max-reconstruction-points", type=int, default=None)
    parser.add_argument("--max-metric-points", type=int, default=None)
    parser.add_argument("--alignment", choices=("identity", "sim3_icp"), default=None)
    parser.add_argument("--fscore-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--coverage-radius", type=float, default=None)
    parser.add_argument("--voxel-downsample-size", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--min-held-out-candidates", type=int, default=None)
    parser.add_argument(
        "--reuse-reconstructions",
        dest="reuse_reconstructions",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-reuse-reconstructions",
        dest="reuse_reconstructions",
        action="store_false",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    return parser.parse_args()


def config_value(
    args: argparse.Namespace, config: dict[str, object], name: str, default: object
) -> object:
    cli_value = getattr(args, name.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def resolve_paths(value: object, label: str) -> list[Path]:
    if not value:
        raise ValueError(f"{label} must be provided")
    raw_paths = [value] if isinstance(value, str | Path) else list(value)
    paths = [Path(path).expanduser().resolve() for path in raw_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: " + ", ".join(map(str, missing)))
    return paths


def resolve_path(value: object, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} must be provided")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return float(value)


def candidate_identifier(value: object) -> str:
    if isinstance(value, int):
        return f"{value:05d}"
    text = str(value)
    path = Path(text)
    return path.stem if path.suffix else text


def normalize_sanity_checks(value: object) -> dict[str, set[str]]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise TypeError("sanity-checks must be a mapping from tag to view IDs/paths")

    normalized: dict[str, set[str]] = {}
    for tag, members in value.items():
        if isinstance(members, str | Path):
            raw_members = [members]
        else:
            raw_members = list(members)
        normalized_tag = "duplicate_input_sensitivity" if str(tag) == "repeat_observed" else str(tag)
        normalized[normalized_tag] = {candidate_identifier(member) for member in raw_members}
    return normalized


def candidate_sanity_tags(
    candidate_path: Path,
    candidate_view_id: str,
    observed_view_ids: set[str],
    sanity_checks: dict[str, set[str]],
) -> list[str]:
    tags = [
        tag
        for tag, members in sanity_checks.items()
        if candidate_view_id in members or str(candidate_path) in members or candidate_path.name in members
    ]
    if candidate_view_id in observed_view_ids and "duplicate_input_sensitivity" not in tags:
        tags.append("duplicate_input_sensitivity")
    return tags


def camera_center_from_pose(pose: list[list[float]]) -> torch.Tensor:
    return torch.tensor([pose[0][3], pose[1][3], pose[2][3]], dtype=torch.float32)


def min_pose_distance_to_observed(
    candidate_pose: list[list[float]], observed_poses: list[list[list[float]]]
) -> float:
    candidate_center = camera_center_from_pose(candidate_pose)
    observed_centers = torch.stack([camera_center_from_pose(pose) for pose in observed_poses])
    return torch.linalg.norm(observed_centers - candidate_center, dim=1).min().item()


def sample_flat_points(
    points: torch.Tensor, confidence: torch.Tensor, max_points: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    points = points.cpu()
    confidence = confidence.cpu()
    if points.shape[0] <= max_points:
        return points, confidence
    indices = torch.randperm(points.shape[0], generator=generator)[:max_points]
    return points[indices], confidence[indices]


def reconstruct_points(
    model: torch.nn.Module,
    image_paths: list[Path],
    checkpoint_path: Path,
    output_dir: Path,
    preprocess_mode: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    layer_index: int,
    min_confidence: float,
    max_points: int,
    seed: int,
    reuse_existing: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "points.pt"
    metadata_path = output_dir / "metadata.json"
    cache_identity = build_reconstruction_cache_identity(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        preprocess_mode=preprocess_mode,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_points,
        seed=seed,
    )
    if reuse_existing and points_path.is_file() and metadata_path.is_file():
        points = torch.load(points_path, map_location="cpu", weights_only=True).float()
        metadata = json.loads(metadata_path.read_text())
        actual_fingerprint = metadata.get("cache_fingerprint")
        if actual_fingerprint != cache_identity["fingerprint"]:
            raise RuntimeError(
                "Refusing to reuse reconstruction cache with mismatched fingerprint: "
                f"expected {cache_identity['fingerprint']}, got {actual_fingerprint}"
            )
        metadata["reused_existing_reconstruction"] = True
        return points, metadata
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=preprocess_mode).to(device)
    extractor = VGGTFeatureExtractor(model, layer_index=layer_index)
    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        features = extractor.extract(images)

    raw_points = features.world_points.detach().reshape(-1, 3).float().cpu()
    raw_confidence = features.world_points_conf.detach().reshape(-1).float().cpu()
    valid = (
        torch.isfinite(raw_points).all(dim=-1)
        & torch.isfinite(raw_confidence)
        & (raw_confidence > min_confidence)
    )
    points = raw_points[valid]
    confidence = raw_confidence[valid]
    points, confidence = sample_flat_points(
        points, confidence, max_points=max_points, seed=seed
    )

    torch.save(points.contiguous(), points_path)
    torch.save(confidence.contiguous(), output_dir / "confidence.pt")
    pose_enc_path = None
    if features.pose_enc is not None:
        pose_enc_path = output_dir / "pose_enc.pt"
        torch.save(features.pose_enc.detach().float().cpu().contiguous(), pose_enc_path)
    metadata = {
        "cache_schema_version": cache_identity["schema_version"],
        "cache_fingerprint": cache_identity["fingerprint"],
        "cache_fingerprint_payload": cache_identity["payload"],
        "image_paths": [str(path) for path in image_paths],
        "pose_enc_path": None if pose_enc_path is None else str(pose_enc_path),
        "input_shape": list(images.shape),
        "patch_grid": list(features.patch_grid),
        "patch_start_idx": features.patch_start_idx,
        "aggregator_forward_count": features.aggregator_forward_count,
        "layer_index": features.layer_index,
        "min_world_point_confidence": min_confidence,
        "finite_world_point_ratio_before_filter": torch.isfinite(raw_points).all(dim=-1).float().mean().item(),
        "valid_world_point_ratio": valid.float().mean().item(),
        "point_count": int(points.shape[0]),
        "confidence_min": confidence.min().item() if confidence.numel() else 0.0,
        "confidence_median": confidence.median().item() if confidence.numel() else 0.0,
        "confidence_max": confidence.max().item() if confidence.numel() else 0.0,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return points, metadata


def maybe_align(points: torch.Tensor, target: torch.Tensor, alignment: str, seed: int) -> torch.Tensor:
    if alignment == "identity":
        return points
    return align_sim3_icp(points, target, seed=seed)


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config(args.config) if args.config else {}

    scene_id = str(config_value(args, config, "scene-id", ""))
    if not scene_id:
        raise ValueError("scene-id must be provided")
    observed_views = resolve_paths(config_value(args, config, "observed-views", None), "observed-views")
    candidate_views = resolve_paths(config_value(args, config, "candidate-views", None), "candidate-views")
    target_points_path = resolve_path(config_value(args, config, "target-points", None), "target-points")
    checkpoint_path = resolve_path(config_value(args, config, "checkpoint", None), "checkpoint")
    output_dir = Path(config_value(args, config, "output-dir", "outputs/oracle_gain/example")).expanduser().resolve()
    preprocess_mode = str(config_value(args, config, "preprocess-mode", "crop"))
    layer_index = int(config_value(args, config, "layer-index", 23))
    min_confidence = float(config_value(args, config, "min-world-point-confidence", 0.0))
    max_reconstruction_points = int(config_value(args, config, "max-reconstruction-points", 50000))
    max_metric_points = int(config_value(args, config, "max-metric-points", 12000))
    alignment = str(config_value(args, config, "alignment", "sim3_icp"))
    thresholds = tuple(
        float(value)
        for value in config_value(args, config, "fscore-thresholds", [0.02, 0.05, 0.1])
    )
    coverage_radius = float(config_value(args, config, "coverage-radius", 0.05))
    voxel_size = optional_float(config_value(args, config, "voxel-downsample-size", None))
    random_seed = int(config_value(args, config, "random-seed", 0))
    min_held_out_candidates = int(config_value(args, config, "min-held-out-candidates", 0))
    reuse_reconstructions = bool(config_value(args, config, "reuse-reconstructions", False))
    split = args.split or str(config.get("split") or scene_split(scene_id))
    point_stride_value = config_value(args, config, "point-stride", None)
    point_stride = None if point_stride_value is None else int(point_stride_value)
    sanity_checks = normalize_sanity_checks(config.get("sanity-checks", {}))

    runtime_config = dict(config)
    if args.device is not None:
        runtime_config["device"] = args.device
    if args.compute_dtype is not None:
        runtime_config["compute_dtype"] = args.compute_dtype
    device = resolve_device(runtime_config)
    compute_dtype = resolve_compute_dtype(runtime_config, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    observed_view_ids = [view_id_from_path(path) for path in observed_views]
    observed_view_id_set = set(observed_view_ids)
    candidate_view_ids = [view_id_from_path(path) for path in candidate_views]
    held_out_candidate_count = sum(
        candidate_id not in observed_view_id_set for candidate_id in candidate_view_ids
    )
    if held_out_candidate_count < min_held_out_candidates:
        raise ValueError(
            f"Expected at least {min_held_out_candidates} held-out candidates, "
            f"received {held_out_candidate_count}"
        )

    observed_poses = [load_pose_matrix(path.with_suffix(".txt")) for path in observed_views]
    memory_id = build_memory_id(scene_id, observed_view_ids)
    print(f"Loading target points from {target_points_path}...")
    target_points = load_point_cloud(target_points_path, point_stride=point_stride)
    print(f"Loading VGGT checkpoint {checkpoint_path} on {device}...")
    model = load_vggt_model("facebook/VGGT-1B", checkpoint_path).to(device)
    model.eval().requires_grad_(False)

    fixed_conditions = {
        "observed_view_ids": observed_view_ids,
        "reference_input_image_order": [str(path) for path in observed_views],
        "candidate_append_after_observed": True,
        "random_seed": random_seed,
        "alignment": alignment,
        "min_world_point_confidence": min_confidence,
        "voxel_downsample_size_meters_after_alignment": voxel_size,
        "max_metric_points": max_metric_points,
        "max_reconstruction_points": max_reconstruction_points,
        "target_point_stride": point_stride,
        "reuse_reconstructions": reuse_reconstructions,
        "preprocess_mode": preprocess_mode,
        "fscore_thresholds_meters_after_alignment": list(thresholds),
        "coverage_radius_meters_after_alignment": coverage_radius,
    }

    reconstruction_root = output_dir / "reconstructions"
    baseline_dir = reconstruction_root / memory_id
    print(f"Reconstructing baseline memory {memory_id} from {len(observed_views)} views...")
    baseline_points, baseline_reconstruction_metadata = reconstruct_points(
        model=model,
        image_paths=observed_views,
        checkpoint_path=checkpoint_path,
        output_dir=baseline_dir,
        preprocess_mode=preprocess_mode,
        device=device,
        compute_dtype=compute_dtype,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_reconstruction_points,
        seed=random_seed,
        reuse_existing=reuse_reconstructions,
    )
    baseline_eval_points = maybe_align(
        baseline_points, target_points, alignment=alignment, seed=random_seed
    )
    baseline_metrics = compute_pointcloud_metrics(
        baseline_eval_points,
        target_points,
        thresholds=thresholds,
        coverage_radius=coverage_radius,
        max_points=max_metric_points,
        voxel_size=voxel_size,
        seed=random_seed,
    )

    records = []
    for index, candidate_path in enumerate(candidate_views):
        candidate_view_id = view_id_from_path(candidate_path)
        pose_path = candidate_path.with_suffix(".txt")
        candidate_pose = load_pose_matrix(pose_path)
        candidate_tags = candidate_sanity_tags(
            candidate_path,
            candidate_view_id,
            observed_view_id_set,
            sanity_checks,
        )
        pose_distance = min_pose_distance_to_observed(candidate_pose, observed_poses)
        candidate_dir = reconstruction_root / f"{memory_id}__plus__{candidate_view_id}"
        print(f"Reconstructing candidate {candidate_view_id} ({index + 1}/{len(candidate_views)})...")
        candidate_points, candidate_reconstruction_metadata = reconstruct_points(
            model=model,
            image_paths=observed_views + [candidate_path],
            checkpoint_path=checkpoint_path,
            output_dir=candidate_dir,
            preprocess_mode=preprocess_mode,
            device=device,
            compute_dtype=compute_dtype,
            layer_index=layer_index,
            min_confidence=min_confidence,
            max_points=max_reconstruction_points,
            seed=random_seed,
            reuse_existing=reuse_reconstructions,
        )
        candidate_eval_points = maybe_align(
            candidate_points, target_points, alignment=alignment, seed=random_seed
        )
        candidate_metrics = compute_pointcloud_metrics(
            candidate_eval_points,
            target_points,
            thresholds=thresholds,
            coverage_radius=coverage_radius,
            max_points=max_metric_points,
            voxel_size=voxel_size,
            seed=random_seed,
        )
        records.append(
            OracleGainRecord(
                scene_id=scene_id,
                split=split,
                memory_id=memory_id,
                observed_view_ids=observed_view_ids,
                candidate_view_id=candidate_view_id,
                candidate_pose=candidate_pose,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                reconstruction_paths={
                    "baseline": str(baseline_dir / "points.pt"),
                    "candidate": str(candidate_dir / "points.pt"),
                },
                metadata={
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "offline_candidate_image_used_for_label": True,
                    "policy_inputs_exclude_candidate_rgb_depth_features_visibility": True,
                    "candidate_sanity_tags": candidate_tags,
                    "pose_min_distance_to_observed_meters": pose_distance,
                    "target_points_path": str(target_points_path),
                    "target_point_count": int(target_points.shape[0]),
                    "metric_units": "meters after Sim(3) alignment to GT" if alignment == "sim3_icp" else "input coordinate units",
                    "fixed_experiment_conditions": fixed_conditions,
                    "preprocess_mode": preprocess_mode,
                    "compute_dtype": str(compute_dtype).removeprefix("torch."),
                    "baseline_reconstruction": baseline_reconstruction_metadata,
                    "candidate_reconstruction": candidate_reconstruction_metadata,
                },
            )
        )

    labels_path = output_dir / "oracle_gain.jsonl"
    write_jsonl(labels_path, records)
    audit_summary = summarize_oracle_audit(records, observed_view_ids=observed_view_ids)
    audit_summary["fixed_experiment_conditions"] = fixed_conditions
    audit_summary["metric_units"] = "meters after Sim(3) alignment to GT" if alignment == "sim3_icp" else "input coordinate units"
    summary_path = output_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_summary, handle, indent=2)
        handle.write("\n")

    run_metadata = {
        "scene_id": scene_id,
        "split": split,
        "memory_id": memory_id,
        "observed_views": [str(path) for path in observed_views],
        "candidate_views": [str(path) for path in candidate_views],
        "target_points": str(target_points_path),
        "checkpoint": str(checkpoint_path),
        "labels_path": str(labels_path),
        "audit_summary_path": str(summary_path),
        "record_count": len(records),
        "held_out_candidate_count": held_out_candidate_count,
        "fixed_experiment_conditions": fixed_conditions,
        "runtime_seconds": time.perf_counter() - started_at,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    run_metadata["output_size_bytes"] = directory_size_bytes(output_dir)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(records)} oracle-gain records to {labels_path}")
    print(json.dumps(run_metadata, indent=2))
    print("Held-out gain summary:")
    for metric_name, stats in audit_summary["held_out_metrics"].items():
        print(metric_name, json.dumps(stats, sort_keys=True))
    print("Sanity checks:")
    for tag, summary in audit_summary["sanity_checks"].items():
        print(tag, json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
