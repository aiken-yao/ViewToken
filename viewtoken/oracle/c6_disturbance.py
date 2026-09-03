"""Stage C6 append-only versus joint-recompute geometry helpers."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from .depth_branch import depth_views_to_known_world_points
from .known_pose import local_camera_points_to_known_world, predicted_world_to_local_camera_points
from .metrics import nearest_neighbor_squared_distances, sample_points
from .v4_branches import V4CacheData, estimate_observed_anchor_alignment_for_cache


def recover_branch_points_by_view(cache: V4CacheData, poses: Tensor, observed_ids: list[str], branch: str, depth_scale: float | None = None) -> tuple[list[Tensor], dict[str, Any]]:
    branch = branch.upper()
    if branch not in {"B", "C", "D"}:
        raise ValueError("C6 supports branches B/C/D")
    poses = torch.as_tensor(poses, dtype=torch.float32).cpu()
    alignment = estimate_observed_anchor_alignment_for_cache(cache.image_view_ids, cache.world_to_camera_extrinsics, poses, observed_ids)
    scale = float(alignment.transform.scale if depth_scale is None else depth_scale)
    if branch == "B":
        local = predicted_world_to_local_camera_points(cache.per_view_world_points, cache.world_to_camera_extrinsics)
        world = local_camera_points_to_known_world(local, poses, depth_scale=scale)
        valid = torch.isfinite(world).all(dim=-1)
    else:
        intrinsics = cache.predicted_intrinsics if branch == "C" else cache.transformed_gt_intrinsics
        world, valid = depth_views_to_known_world_points(cache.depth, intrinsics, poses, depth_scale=scale)
        valid = valid & torch.isfinite(world).all(dim=-1)
    return [world[i].reshape(-1, 3)[valid[i].reshape(-1)].contiguous() for i in range(cache.view_count)], {"branch": branch, "depth_scale": scale, "per_cache_depth_scale": float(alignment.transform.scale)}


def sample_views(points_by_view: list[Tensor], quota: int, seed: int) -> list[Tensor]:
    return [sample_points(points, quota, seed=seed + 1009 * i, method="hash") for i, points in enumerate(points_by_view)]


def build_disturbance_states(baseline_views: list[Tensor], recomputed_views: list[Tensor], quota: int, seed: int) -> dict[str, Tensor]:
    base = sample_views(baseline_views, quota, seed)
    recomputed = sample_views(recomputed_views, quota, seed)
    h0 = torch.cat(base, dim=0)
    h1 = torch.cat(recomputed[:-1], dim=0)
    h2 = torch.cat(base + [recomputed[-1]], dim=0)
    h3 = torch.cat(recomputed, dim=0)
    if not torch.equal(h2[: h0.shape[0]], h0):
        raise AssertionError("H2 must preserve H0 points and order exactly")
    return {"H0": h0, "H1": h1, "H2": h2, "H3": h3}


def state_metrics(target: Tensor, states: dict[str, Tensor], observed_mask: Tensor, novel_mask: Tensor, thresholds=(0.05, 0.10, 0.20), chunk_size=2048) -> dict[str, Any]:
    target = torch.as_tensor(target, dtype=torch.float32).cpu()
    masks = {"observed": observed_mask.bool().cpu(), "novel": novel_mask.bool().cpu()}
    result: dict[str, Any] = {}
    for name, points in states.items():
        distances = nearest_neighbor_squared_distances(target, points, chunk_size=chunk_size).sqrt()
        state = {}
        for region, mask in masks.items():
            values = distances[mask]
            covered = {f"{t:g}": {"count": int((values <= t).sum()), "ratio": None if values.numel() == 0 else float((values <= t).float().mean())} for t in thresholds}
            state[region] = {"target_count": int(values.numel()), "covered": covered, "median": None if values.numel() == 0 else float(values.median()), "p90": None if values.numel() == 0 else float(torch.quantile(values, .9))}
        result[name] = state
    effects = {}
    for region in masks:
        effects[region] = {}
        for t in thresholds:
            key=f"{t:g}"; h={n: result[n][region]["covered"][key]["count"] for n in states}
            effects[region][key] = {"candidate_addition": h["H2"]-h["H0"], "history_recompute": h["H3"]-h["H2"], "joint_net": h["H3"]-h["H0"], "observed_recompute": h["H1"]-h["H0"]}
    return {"states": result, "effects": effects}


def append_only_coverage_is_monotonic(target: Tensor, baseline: Tensor, added: Tensor, thresholds=(0.05, .10, .20)) -> bool:
    before = nearest_neighbor_squared_distances(target, baseline).sqrt()
    after = nearest_neighbor_squared_distances(target, torch.cat([baseline, added])).sqrt()
    return all(int((after <= t).sum()) >= int((before <= t).sum()) for t in thresholds)
