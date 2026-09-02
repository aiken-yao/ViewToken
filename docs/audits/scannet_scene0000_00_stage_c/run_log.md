# Stage C Deterministic Smoke Audit

Status: `complete`, but calibration is still not passed.

This run keeps the Stage B camera-anchor protocol, regenerates the same small smoke set as complete v3 caches, removes the pre-alignment random 50k reconstruction truncation, and uses deterministic hash sampling for metric evaluation. It did not run the full audit20 set.

## Inputs

- Observed views: `00000`, `00010`, `00020`
- Smoke candidates: `00010`, `00019`, `00325`, `00425`
- Reconstruction output: `outputs/oracle_gain/scannet_scene0000_00_stage_c_deterministic_smoke_v3/`
- Camera-anchor output: `outputs/oracle_calibration/scannet_scene0000_00_stage_c_deterministic_smoke_v3/`
- Python: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python`
- Checkpoint: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/model_weights/vggt/model.pt`

## Cache Policy

- `max_reconstruction_points`: `null`
- `reconstruction_sample_method`: `none`
- `max_metric_points`: `12000`
- `metric_sample_method`: `hash`
- `voxel_downsample_size`: `0.02`
- `reuse_reconstructions`: `false`

## Runtime

- Generator runtime seconds: `57.941`
- Generator peak GPU memory bytes: `6158511104`
- Output size bytes: `61806190`
- Stage C camera-anchor runtime seconds: `45.610`

## Reconstruction Cache Counts

- `23a55d429693f934`: input_shape `[3, 3, 392, 518]`, raw `609168`, filtered `609168`, saved `609168`, sample `none`
- `23a55d429693f934__plus__00010`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00019`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00325`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00425`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`

## Held-Out Gain Summary

| metric | mean gain | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---|---:|
| `chamfer` | `0.41704532` | `1.000` | `00325` | `0.74176456` |
| `accuracy` | `-0.12213880` | `0.000` | `00019` | `-0.00679053` |
| `completeness` | `0.95622945` | `1.000` | `00325` | `1.78152227` |
| `coverage` | `-0.00127778` | `0.333` | `00425` | `0.00050000` |
| `fscore@0.02` | `-0.00234810` | `0.000` | `00425` | `-0.00141536` |
| `fscore@0.05` | `-0.00343737` | `0.000` | `00425` | `-0.00007769` |
| `fscore@0.1` | `-0.00609095` | `0.000` | `00425` | `-0.00138420` |

## Point-Cloud Diagnostics

| protocol | residual mean | residual RMSE | inlier@0.05m |
|---|---:|---:|---:|
| `baseline_camera_anchor_point_residuals` | `0.156149` | `0.200617` | `0.087667` |
| `candidate_camera_anchor_point_residuals` | `0.247632` | `0.322040` | `0.072021` |
| `baseline_old_free_icp_point_residuals` | `0.526998` | `0.642022` | `0.025750` |
| `candidate_old_free_icp_point_residuals` | `0.443817` | `0.548059` | `0.037042` |

## Per-Candidate Camera-Anchor Results

| candidate | tags | candidate scale | candidate camera RMSE | chamfer gain | accuracy gain | completeness gain | coverage gain | F@0.05 gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `00010` | `duplicate_input_sensitivity` | `1.948739` | `0.012537` | `0.015020` | `0.000485` | `0.029555` | `0.000500` | `0.001003` |
| `00019` | `high_overlap_neighbor` | `1.938811` | `0.013129` | `0.020601` | `-0.006791` | `0.047993` | `-0.000917` | `-0.001769` |
| `00325` | `old_oracle_best` | `2.163518` | `0.015898` | `0.741765` | `-0.297993` | `1.781522` | `-0.003417` | `-0.008465` |
| `00425` | `new_area` | `2.161189` | `0.016536` | `0.488770` | `-0.061633` | `1.039173` | `0.000500` | `-0.000078` |

## Interpretation

Camera-anchor alignment remains much better than old free-ICP on point residuals, and all v3 caches are complete. Removing the random pre-alignment reconstruction truncation does not make coverage or F-score reliably positive. The remaining issue is therefore more likely in the evaluation target/visibility protocol, candidate semantic selection, or VGGT point outliers than in the old 50k truncation alone.

Do not expand to full audit20 or train policy from this result. The next calibration step should inspect GT frustum overlap and novel visible surface fraction for these candidates, especially `00325` and `00425`, before deciding whether they are valid high-value new-area cases.
