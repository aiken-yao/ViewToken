# Stage C3 Known-Pose Diagnostic

Status: `complete`

Stage C3 passed: `False`

This diagnostic reused existing complete v3 caches and did not run VGGT.

## Scope

- Predicted-world branch: VGGT `world_points` plus observed-camera-anchor Sim(3).
- Known-pose branch: cached v3 `world_points` converted back to per-view local camera coordinates with VGGT predicted extrinsics, scaled by observed-anchor Sim(3), then fused with ScanNet GT camera-to-world poses.
- True v4 depth-backprojection branch status: `blocked_missing_v4_depth_conf_per_view_offsets_and_preprocessing_transform; cached v3 diagnostic used local camera coordinates recovered from world_points and pose_enc instead`
- Candidate RGB/depth/visibility remains offline audit data only, not future policy input.

## Candidate Summary

| cand | corrected visibility tag | cand overlap frac | novel scene frac | held-out center err m | rot err deg | pairwise err mean/max m | A cand covered novel @0.05/@0.10 | B cand covered novel @0.05/@0.10 | A obs retention@0.05 | B obs retention@0.05 | B global acc gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `00010` | `duplicate_or_no_novel` | `1.000000` | `0.000000` | `0.0048` | `19.69` | `0.0089`/`0.0210` | `0`/`0` | `0`/`0` | `0.003060` | `-0.002295` | `0.001607` |
| `00019` | `high_overlap_low_novel` | `0.998900` | `0.000083` | `0.0146` | `20.21` | `0.0106`/`0.0217` | `0`/`0` | `0`/`0` | `-0.009946` | `-0.042081` | `-0.013102` |
| `00325` | `disconnected_novel_view` | `0.000000` | `0.236500` | `2.5218` | `21.38` | `1.1891`/`2.5246` | `0`/`0` | `331`/`787` | `-0.033665` | `-0.068095` | `-0.008211` |
| `00425` | `disconnected_novel_view` | `0.000000` | `0.027083` | `4.7935` | `42.92` | `1.5329`/`3.2760` | `0`/`0` | `0`/`0` | `-0.007651` | `-0.003826` | `-0.011641` |

## C2 Semantic Correction

- `00325` and `00425` have `M_overlap = 0` in the nominal C2 masks, so they are `disconnected_novel_view`, not connected-new-area views.
- A connected-novel candidate must have both `novel_count > 0` and `overlap_count > 0`.
- Overlap stability is recorded across `27` C2 variants in the JSON report.

## Assessment

- Predicted-world connected candidates with stable positive novel gain: `[]`
- Known-pose connected candidates with stable positive novel gain: `[]`
- Disconnected novel candidates: `['00325', '00425']`
- Known-pose branch supported by existing v3 cache: `True`

Full JSON summary: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c3/stage_c3_known_pose_summary.json`
