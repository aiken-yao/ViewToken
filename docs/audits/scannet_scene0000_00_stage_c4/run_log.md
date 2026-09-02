# Stage C4 Connected-Depth Diagnostics

Status: `complete`

This run did not execute VGGT, did not expand audit20, and did not train a policy.

## Candidate Mining

- Scene pose candidates scanned: `597`
- Connected-novel RGB candidates found: `190`
- Connected-novel definition: `novel_count > 0 && overlap_count > 0`
- Visibility source: `ScanNet GT point z-buffer over full 50k target points; no dense depth/mesh visibility is used. C4 adds pixel_radius splatting stability checks to expose sparse-point sensitivity.`
- Nominal visibility: depth tolerance `0.05`, pixel radius `0`

## Fixed Selection

- `high_overlap_low_novel`: `00018` (overlap >= connected p75 and novel_scene <= connected p25; sort by overlap desc, novelty asc, id asc)
- `medium_overlap_medium_novel`: `00369` (overlap and novel_scene both within connected p25-p75; sort by distance to connected medians, id asc)
- `high_novel_connected`: `00332` (novel_scene >= connected p90; sort by novelty desc, overlap desc, id asc)
- `nearest_connected`: `00019` (minimum distance to any observed camera among connected-novel RGB poses)
- `farthest_connected`: `00432` (maximum distance to any observed camera among connected-novel RGB poses)

Selected unique candidates: `['00018', '00369', '00332', '00019', '00432']`
Expected v4 cache count after approval: `6`

## Top Connected By Novelty

| cand | cand overlap frac | novel scene frac | visible scene frac | min dist m | min angle deg |
|---|---:|---:|---:|---:|---:|
| `00332` | `0.000383` | `0.208940` | `0.209020` | `3.1023` | `44.78` |
| `00082` | `0.000259` | `0.154440` | `0.154480` | `1.0062` | `59.72` |
| `00081` | `0.000265` | `0.150740` | `0.150780` | `1.0060` | `59.48` |
| `00080` | `0.001405` | `0.142120` | `0.142320` | `0.9831` | `58.10` |
| `00065` | `0.134689` | `0.122580` | `0.141660` | `0.1476` | `51.77` |
| `00071` | `0.025223` | `0.122120` | `0.125280` | `0.1890` | `51.13` |
| `00074` | `0.025834` | `0.121420` | `0.124640` | `0.4227` | `45.92` |
| `00437` | `0.301980` | `0.121260` | `0.173720` | `4.1884` | `3.53` |
| `00438` | `0.299084` | `0.120880` | `0.172460` | `4.1903` | `3.61` |
| `00073` | `0.022366` | `0.120640` | `0.123400` | `0.4039` | `46.06` |
| `00079` | `0.037114` | `0.119860` | `0.124480` | `0.8399` | `53.07` |
| `00439` | `0.303245` | `0.119800` | `0.171940` | `4.1906` | `4.72` |

## Top Connected By Overlap

| cand | cand overlap frac | novel scene frac | visible scene frac | min dist m | min angle deg |
|---|---:|---:|---:|---:|---:|
| `00018` | `0.997855` | `0.000160` | `0.074600` | `0.0084` | `1.00` |
| `00017` | `0.996531` | `0.000260` | `0.074940` | `0.0580` | `1.02` |
| `00019` | `0.996298` | `0.000280` | `0.075640` | `0.0082` | `0.24` |
| `00009` | `0.986966` | `0.000860` | `0.065980` | `0.0351` | `4.15` |
| `00016` | `0.984384` | `0.001100` | `0.070440` | `0.2025` | `6.05` |
| `00008` | `0.984309` | `0.001040` | `0.066280` | `0.0656` | `4.62` |
| `00015` | `0.982664` | `0.001200` | `0.069220` | `0.1989` | `6.39` |
| `00596` | `0.981518` | `0.001520` | `0.082240` | `0.1895` | `6.52` |
| `00262` | `0.981044` | `0.001380` | `0.072800` | `0.3713` | `2.54` |
| `00014` | `0.980450` | `0.001320` | `0.067520` | `0.1980` | `7.32` |
| `00007` | `0.971818` | `0.001820` | `0.064580` | `0.0641` | `8.17` |
| `00013` | `0.971334` | `0.001900` | `0.066280` | `0.2025` | `7.57` |

## Splatting Stability

Stability variants: `84`

JSON summary contains per-candidate overlap/novel distributions for point limits `[12000, 50000]`, depth tolerances `[0.02, 0.05, 0.1]`, and pixel radii `[0, 1]`.

## 00425 v3 Diagnosis

- Candidate-view confidence median/min/max: `1.0` / `1.0` / `1.0000067949295044`
- Point-head local Z median/p10/p90: `-0.21513012051582336` / `-0.3369552493095398` / `-0.1396857351064682`
- GT novel local Z median/p10/p90: `1.9596922397613525` / `1.666896104812622` / `2.1897945404052734`
- Ray pixel error median with predicted intrinsics: `None`
- Ray pixel error median with transformed calibrated intrinsics: `None`
- Likely primary failure: `point_head_local_geometry_behind_candidate_camera`
- Low-confidence failure: `False`

- GT visibility marks 00425 as disconnected novel: novel_count > 0 and overlap_count == 0
- point-head local Z median is <= 0 while GT novel Z median is > 0
- candidate-view confidence is nearly constant, so quantile filtering cannot separate good geometry
- confidence sweep recovered zero GT novel points at 0.05m and 0.10m
- ray/intrinsics checks have zero positive-Z local points to project

| conf quantile | threshold | kept points | novel covered @0.05 | novel covered @0.10 | outlier ratio |
|---:|---:|---:|---:|---:|---:|
| `0.0` | `1.000000` | `203056` | `0` | `0` | `0.366100` |
| `0.1` | `1.000000` | `203056` | `0` | `0` | `0.367233` |
| `0.25` | `1.000000` | `203056` | `0` | `0` | `0.366100` |
| `0.5` | `1.000000` | `203056` | `0` | `0` | `0.366267` |
| `0.75` | `1.000000` | `203056` | `0` | `0` | `0.366533` |
| `0.9` | `1.000000` | `203056` | `0` | `0` | `0.369400` |
| `0.95` | `1.000000` | `203056` | `0` | `0` | `0.364733` |

## Branch Status

- A predicted-world: loaded from Stage C3 existing v3 cache when available.
- B point-head known-pose: loaded from Stage C3 and diagnosed here from candidate-view local geometry.
- C depth + predicted intrinsics + known pose: implemented in `viewtoken/oracle/depth_branch.py`, not run because no v4 cache exists yet.
- D depth + transformed calibrated intrinsics + known pose: implemented in `viewtoken/oracle/depth_branch.py`, not run because no v4 cache exists yet.

## Proposed Commands

```bash
/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python scripts/generate_oracle_gain.py --config /mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c4/proposed_v4_connected_candidates.yaml
```

```bash
/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python scripts/audit_stage_c4_connected_depth.py --config configs/oracle_stage_c4_connected_depth.yaml
```

Full JSON summary: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c4/stage_c4_connected_depth_summary.json`
Candidate records: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_calibration/scannet_scene0000_00_stage_c4/stage_c4_candidate_visibility_records.jsonl`
Candidate records for GitHub: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c4/stage_c4_candidate_visibility_records.jsonl`
Stability records: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_calibration/scannet_scene0000_00_stage_c4/stage_c4_splatting_stability_records.jsonl`
Stability records for GitHub: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c4/stage_c4_splatting_stability_records.jsonl`
Proposed v4 config: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c4/proposed_v4_connected_candidates.yaml`

## Validation

- `pytest tests -q`: `64 passed`
- `python -m compileall viewtoken scripts tests`: `passed`
- `git diff --check`: `passed`
