#!/usr/bin/env python3
"""Cache-only Stage C5 audit for VGGT branches A/B/C/D."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_vggt_features import load_config  # noqa: E402
from viewtoken.oracle import (  # noqa: E402
    PinholeIntrinsics,
    build_per_view_shape_offsets,
    build_memory_id,
    build_v4_reconstruction_cache_identity,
    build_visibility_masks,
    cache_artifact_shape_summary,
    candidate_view_depth_diagnostics,
    compare_branch_reconstructions,
    compute_batch_preprocess_transforms,
    heldout_candidate_pose_diagnostics,
    load_gt_poses_for_view_ids,
    load_point_cloud,
    load_pose_matrix,
    load_v4_cache_data,
    recover_v4_branch_points,
    summarize_visibility_masks,
)
from viewtoken.oracle.metrics import sample_points  # noqa: E402

BRANCHES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/oracle_stage_c5_v4_smoke.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ids_from_paths(values: list[str]) -> list[str]:
    return [Path(value).stem for value in values]


def expected_cache_dir(root: Path, memory_id: str, candidate_id: str | None = None) -> Path:
    return root / memory_id if candidate_id is None else root / f"{memory_id}__plus__{candidate_id}"


def load_expected_cache(path: Path, image_paths: list[Path], config: dict):
    sizes = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.append((image.width, image.height))
    transforms = compute_batch_preprocess_transforms(
        sizes,
        mode=str(config.get("preprocess-mode", "crop")),
        target_size=518,
        patch_size=14,
    )
    offsets = build_per_view_shape_offsets(
        [image_path.stem for image_path in image_paths],
        height=transforms[0].output_height,
        width=transforms[0].output_width,
    )
    calibrated_matrix = torch.tensor(
        np.loadtxt(config["calibrated-intrinsics"], dtype=np.float32)
    )[:3, :3]
    identity = build_v4_reconstruction_cache_identity(
        checkpoint_path=Path(config["checkpoint"]),
        image_paths=image_paths,
        preprocess_mode=str(config.get("preprocess-mode", "crop")),
        layer_index=int(config.get("layer-index", 23)),
        min_confidence=float(config.get("min-world-point-confidence", 0.0)),
        max_points=None,
        seed=int(config.get("random-seed", 0)),
        sample_method=str(config.get("reconstruction-sample-method", "none")),
        preprocessing_transforms=[transform.to_dict() for transform in transforms],
        per_view_shape_offsets=offsets,
        calibrated_intrinsics=calibrated_matrix.tolist(),
    )
    return load_v4_cache_data(
        path,
        expected_fingerprint=str(identity["fingerprint"]),
        expected_view_ids=[image_path.stem for image_path in image_paths],
    )


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Stage C5 v4 Branch Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Validated caches: `{summary.get('validated_cache_count', 0)}` / `{summary.get('expected_cache_count', 6)}`",
    ]
    if summary.get("error"):
        lines += ["", "## Blocker", "", f"`{summary['error']}`"]
    for row in summary.get("candidate_rows", []):
        lines += ["", f"## Candidate {row['candidate_view_id']}", ""]
        lines.append(f"Visibility: `{json.dumps(row['visibility_stats'], sort_keys=True)}`")
        for branch, payload in row["branches"].items():
            novel = payload["comparison"]["novel"]["covered"]
            observed = payload["comparison"]["observed"]["covered"]
            lines.append(
                f"- {branch}: novel gain @.05/.10 = "
                f"{novel['0.05']['covered_count_gain']}/{novel['0.1']['covered_count_gain']}; "
                f"observed gain @.05/.10 = "
                f"{observed['0.05']['covered_count_gain']}/{observed['0.1']['covered_count_gain']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def connectivity_preflight(config: dict, observed_ids: list[str], candidate_ids: list[str]) -> dict:
    pose_dir = Path(config["posed-image-dir"])
    source = load_point_cloud(
        Path(config["target-points"]), point_stride=int(config.get("point-stride", 6))
    )
    matrix = torch.tensor(np.loadtxt(config["intrinsics"], dtype=np.float32))[:3, :3]
    with Image.open(config["observed-views"][0]) as image:
        intrinsics = PinholeIntrinsics.from_matrix(matrix, image.width, image.height)
    observed_poses = [
        torch.tensor(load_pose_matrix(pose_dir / f"{view_id}.txt"), dtype=torch.float32)
        for view_id in observed_ids
    ]
    records = []
    for point_limit in config.get("stability-point-limits", [12000, 50000]):
        points = sample_points(source, int(point_limit), seed=0, method="hash")
        for tolerance in config.get("stability-depth-tolerances", [0.02, 0.05, 0.10]):
            for radius in config.get("stability-pixel-radii", [0, 1]):
                for candidate_id in candidate_ids:
                    candidate_pose = torch.tensor(
                        load_pose_matrix(pose_dir / f"{candidate_id}.txt"), dtype=torch.float32
                    )
                    masks = build_visibility_masks(
                        points,
                        observed_poses,
                        candidate_pose,
                        intrinsics,
                        depth_tolerance=float(tolerance),
                        pixel_radius=int(radius),
                    )
                    records.append(
                        {
                            "candidate_view_id": candidate_id,
                            "point_limit": int(point_limit),
                            "depth_tolerance": float(tolerance),
                            "pixel_radius": int(radius),
                            "visibility_stats": summarize_visibility_masks(masks).to_dict(),
                        }
                    )
    summary = {}
    for candidate_id in candidate_ids:
        rows = [row for row in records if row["candidate_view_id"] == candidate_id]
        overlaps = [int(row["visibility_stats"]["overlap_count"]) for row in rows]
        fractions = [float(row["visibility_stats"]["candidate_overlap_fraction"] or 0.0) for row in rows]
        summary[candidate_id] = {
            "variant_count": len(rows),
            "min_overlap_count": min(overlaps),
            "max_overlap_count": max(overlaps),
            "min_candidate_overlap_fraction": min(fractions),
            "always_nonzero_overlap": min(overlaps) > 0,
        }
    unstable = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id != "00018" and not summary[candidate_id]["always_nonzero_overlap"]
    ]
    return {"summary_by_candidate": summary, "records": records, "unstable_non_control_candidates": unstable}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    scene_id = str(config["scene-id"])
    observed_ids = ids_from_paths(config["observed-views"])
    candidate_ids = ids_from_paths(config["candidate-views"])
    output_dir = Path(config["output-dir"])
    docs_dir = Path(config.get("docs-output-dir", "docs/audits/stage_c5"))
    result_path = args.output or docs_dir / "stage_c5_v4_branch_summary.json"
    run_log_path = docs_dir / "run_log.md"
    cache_root = output_dir / "reconstructions"
    memory_id = build_memory_id(scene_id, observed_ids)
    expected = [
        ("baseline", expected_cache_dir(cache_root, memory_id), observed_ids),
        *[
            (candidate_id, expected_cache_dir(cache_root, memory_id, candidate_id), observed_ids + [candidate_id])
            for candidate_id in candidate_ids
        ],
    ]
    preflight = connectivity_preflight(config, observed_ids, candidate_ids)
    preflight_path = docs_dir / "stage_c5_connectivity_preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if preflight["unstable_non_control_candidates"]:
        summary = {
            "stage": "C5",
            "status": "blocked_unstable_candidate_connectivity",
            "expected_cache_count": len(expected),
            "validated_cache_count": 0,
            "connectivity_preflight": preflight["summary_by_candidate"],
            "unstable_non_control_candidates": preflight["unstable_non_control_candidates"],
            "did_run_vggt": False,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_markdown(run_log_path, summary)
        print(json.dumps(summary, indent=2))
        return 2
    missing = [
        {"label": label, "path": str(path), "expected_view_ids": view_ids}
        for label, path, view_ids in expected
        if not path.is_dir()
    ]
    if missing:
        summary = {
            "stage": "C5",
            "status": "blocked_missing_v4_cache",
            "expected_cache_count": len(expected),
            "validated_cache_count": 0,
            "missing": missing,
            "connectivity_preflight": preflight["summary_by_candidate"],
            "did_run_vggt": False,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_markdown(run_log_path, summary)
        print(json.dumps(summary, indent=2))
        return 2

    try:
        observed_paths = [Path(value) for value in config["observed-views"]]
        candidate_paths = {Path(value).stem: Path(value) for value in config["candidate-views"]}
        baseline = load_expected_cache(expected[0][1], observed_paths, config)
        baseline_poses = load_gt_poses_for_view_ids(observed_ids, Path(config["posed-image-dir"]))
        baseline_points = {
            branch: recover_v4_branch_points(baseline, baseline_poses, observed_ids, branch)
            for branch in BRANCHES
        }
        target = load_point_cloud(Path(config["target-points"]), point_stride=int(config.get("point-stride", 6)))
        target = sample_points(target, int(config.get("max-surface-points", 12000)), seed=0, method="hash")
        matrix = torch.tensor(np.loadtxt(config["intrinsics"], dtype=np.float32))[:3, :3]
        with Image.open(config["observed-views"][0]) as image:
            intrinsics = PinholeIntrinsics.from_matrix(matrix, image.width, image.height)
        pose_dir = Path(config["posed-image-dir"])
        candidate_rows = []
        for candidate_id, candidate_path, view_ids in expected[1:]:
            cache = load_expected_cache(
                candidate_path,
                observed_paths + [candidate_paths[candidate_id]],
                config,
            )
            poses = load_gt_poses_for_view_ids(view_ids, pose_dir)
            masks = build_visibility_masks(
                target,
                [poses[index] for index in range(len(observed_ids))],
                poses[-1],
                intrinsics,
                depth_tolerance=float(config.get("visibility-depth-tolerance", 0.05)),
                pixel_radius=int(config.get("visibility-pixel-radius", 0)),
            )
            branch_rows = {}
            for branch in BRANCHES:
                candidate_points, candidate_recovery = recover_v4_branch_points(
                    cache, poses, observed_ids, branch
                )
                base_points, baseline_recovery = baseline_points[branch]
                branch_rows[branch] = {
                    "baseline_recovery": baseline_recovery,
                    "candidate_recovery": candidate_recovery,
                    "comparison": compare_branch_reconstructions(
                        target,
                        base_points,
                        candidate_points,
                        masks.observed,
                        masks.novel,
                        thresholds=(0.05, 0.10, 0.20),
                    ),
                }
            candidate_rows.append(
                {
                    "candidate_view_id": candidate_id,
                    "visibility_stats": summarize_visibility_masks(masks).to_dict(),
                    "cache": cache_artifact_shape_summary(cache),
                    "depth_diagnostics": candidate_view_depth_diagnostics(cache, candidate_id),
                    "heldout_pose_diagnostics": heldout_candidate_pose_diagnostics(
                        cache, poses, observed_ids, candidate_id
                    ),
                    "branches": branch_rows,
                }
            )
        non_control = [row for row in candidate_rows if row["candidate_view_id"] != "00018"]
        positive_by_branch = {}
        for branch in BRANCHES:
            positive_by_branch[branch] = [
                row["candidate_view_id"]
                for row in non_control
                if row["branches"][branch]["comparison"]["novel"]["covered"]["0.05"]["covered_count_gain"] > 0
                and row["branches"][branch]["comparison"]["novel"]["covered"]["0.1"]["covered_count_gain"] > 0
            ]
        summary = {
            "stage": "C5",
            "status": "complete",
            "expected_cache_count": 6,
            "validated_cache_count": 6,
            "observed_view_ids": observed_ids,
            "candidate_view_ids": candidate_ids,
            "connectivity_preflight": preflight["summary_by_candidate"],
            "baseline_cache": cache_artifact_shape_summary(baseline),
            "candidate_rows": candidate_rows,
            "positive_non_control_candidates_by_branch": positive_by_branch,
            "assessment": {
                "depth_known_pose_smoke_passed": len(set(positive_by_branch["C"] + positive_by_branch["D"])) >= 2,
                "do_not_expand_automatically": True,
            },
            "did_run_vggt": False,
        }
    except Exception as exc:  # fail closed
        summary = {
            "stage": "C5",
            "status": "failed_v4_branch_audit",
            "expected_cache_count": 6,
            "validated_cache_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "did_run_vggt": False,
        }
        exit_code = 3
    else:
        exit_code = 0
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(run_log_path, summary)
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
