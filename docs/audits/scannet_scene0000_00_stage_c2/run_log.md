# Stage C2 GT Visibility Audit

Status: `complete`

Stage C2 passed: `False`

This audit reused the complete Stage C deterministic v3 reconstruction caches and did not run VGGT. Candidate RGB/depth visibility is used only for offline oracle-label audit, not as future policy input.

## Convention

- ScanNet pose convention: `ScanNet 4x4 camera-to-world; translation is camera center in world coordinates.`
- Projection convention: `OpenCV pinhole camera coordinates; +Z is in front; u=fx*x/z+cx, v=fy*y/z+cy; pixel bounds are half-open [0,width)x[0,height).`
- Image size: `1296x968`
- Intrinsics source: `/mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00/intrinsic.txt`
- Occlusion source: `GT point-cloud z-buffer over the fixed deterministic surface sample; no RGB-D depth maps were found in posed_images/scene0000_00.`
- Nominal voxel/depth/seed: `{'gt_voxel_size': 0.02, 'visibility_depth_tolerance': 0.05, 'surface_sample_seed_offset': 0}`
- Synthetic visibility test count: `7`

## Nominal Results

| cand | tags | visible scene frac | cand novel frac | novel scene frac | cand overlap frac | largest novel comp | novel gain@0.05 scene | novel gain@0.10 scene | obs retention@0.05 | obs retention@0.10 | union compl gain | global acc gain | min dist | min angle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `00010` | `duplicate_input_sensitivity` | `0.062250` | `0.000000` | `0.000000` | `1.000000` | `0` | `0.000000` | `0.000000` | `0.003060` | `0.015302` | `0.001229` | `0.000389` | `0.000000` | `0.000` |
| `00019` | `high_overlap_neighbor` | `0.075750` | `0.001100` | `0.000083` | `0.998900` | `1` | `0.000000` | `0.000000` | `-0.009946` | `-0.016832` | `-0.013269` | `-0.008143` | `0.008193` | `0.239` |
| `00325` | `old_oracle_best` | `0.236500` | `1.000000` | `0.236500` | `0.000000` | `40` | `0.000000` | `0.000000` | `-0.033665` | `-0.042846` | `2.164370` | `-0.298661` | `2.358865` | `99.050` |
| `00425` | `new_area` | `0.027083` | `1.000000` | `0.027083` | `0.000000` | `10` | `0.000000` | `0.000000` | `-0.007651` | `0.003060` | `0.002318` | `-0.062003` | `3.914814` | `25.209` |

## Stability

| cand | novel scene mean | novel scene std | gain@0.05 mean | gain@0.05 std | gain@0.10 mean | gain@0.10 std |
|---|---:|---:|---:|---:|---:|---:|
| `00010` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `00019` | `0.000139` | `0.000068` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `00325` | `0.237025` | `0.003940` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `00425` | `0.028981` | `0.001595` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |

## Assessment

- Duplicate novelty near zero: `True`
- High-overlap novelty less than connected new area: `False`
- Connected-new-area candidates with novel surface and nonzero overlap: `[]`
- Disconnected novel candidates: `['00325', '00425']`
- Connected-novel requires overlap: `True`
- Novel connectivity min largest component count: `10`
- Stable positive connected new-area candidates: `[]`
- Observed retention not badly damaged: `False`
- Strict best candidate counts by gain@0.05: `{}`
- Best-candidate tie groups by gain@0.05: `{'00010|00019|00325|00425': 27}`

Full JSON summary: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/docs/audits/scannet_scene0000_00_stage_c2/stage_c2_visibility_summary.json`
Variant records: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_calibration/scannet_scene0000_00_stage_c2/stage_c2_visibility_records.jsonl`
