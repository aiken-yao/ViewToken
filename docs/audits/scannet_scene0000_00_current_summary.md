# ScanNet scene0000_00 Oracle Calibration Summary

Date: 2026-09-02

## Current Verdict

Oracle measurement calibration is **not passed yet**. Camera-anchor alignment and v3 cache integrity are now implemented and the small smoke path runs end-to-end, but the deterministic Stage C smoke still shows negative or unstable coverage/F-score. Do not expand to full audit20 and do not train policy from the current labels.

## Source Outputs

- Original audit20: `outputs/oracle_gain/scannet_scene0000_00_audit20/`
- Stage A: `outputs/oracle_calibration/scannet_scene0000_00_stage_a/`
- Stage B old-cache block: `outputs/oracle_calibration/scannet_scene0000_00_stage_b/`
- Stage B smoke v3: `outputs/oracle_calibration/scannet_scene0000_00_stage_b_smoke_v3/`
- Stage C deterministic smoke v3: `outputs/oracle_calibration/scannet_scene0000_00_stage_c_deterministic_smoke_v3/`

## Original Audit20 Conclusion

The original audit20 labels are not trustworthy for training. The old free Sim(3) ICP alignment could improve accuracy while hurting completeness/coverage, indicating local collapse or poor overlap rather than a reliable NBV label protocol.

Held-out audit20 summary from the previous report:

```text
Chamfer gain mean = -0.1252, positive ratio = 0.05
accuracy gain mean = +0.1300, positive ratio = 0.80
completeness gain mean = -0.3803, positive ratio = 0.05
coverage gain mean = -0.00190, positive ratio = 0.40
F-score@0.02 gain mean = +0.00042, positive ratio = 0.65
F-score@0.05 gain mean = +0.00088, positive ratio = 0.60
F-score@0.10 gain mean = -0.00999, positive ratio = 0.35
```

## Stage A Conclusion

Stage A partially passed but did not clear overall calibration. Identical-cloud arithmetic is stable, but real VGGT partial reconstructions aligned to full ScanNet GT with free ICP have weak correspondence.

```text
identical-cloud max_abs_gain = 0.0
free Sim(3) ICP residual RMSE mean ~= 0.669 m
free Sim(3) ICP inlier ratio @ 0.05m mean ~= 0.00869
```

The key readout is that the metric arithmetic and known Sim(3) tests stand, while old free-ICP alignment remains invalid for trainable labels.

## Stage B Old-Cache Block

Stage B initially blocked correctly on old audit20 caches. The stricter validator now reports incomplete v3 artifacts:

```text
status = blocked_invalid_reconstruction_cache
attempted records = 21
invalid reconstruction caches = 22
missing pose_enc.pt = 22
did_run_vggt = false
```

Reason: old audit20 caches do not have `pose_enc.pt`, schema `oracle-reconstruction-v3`, or cache fingerprint. Do not patch pose encodings from separate VGGT runs into old caches.

## Stage B Smoke v3 Conclusion

After regenerating a small v3 smoke set, camera-anchor alignment completed. This showed that the camera-anchor protocol is operational and much better than old free-ICP on point-cloud residuals, but the label distribution still did not pass because coverage/F-score remained weak.

| metric | mean gain | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---|---:|
| `chamfer` | `0.42012405` | `1.000` | `00325` | `0.78890297` |
| `accuracy` | `-0.09655116` | `0.000` | `00019` | `-0.00775485` |
| `completeness` | `0.93679925` | `1.000` | `00325` | `1.78661239` |
| `coverage` | `-0.00236111` | `0.000` | `00425` | `-0.00091667` |
| `fscore@0.02` | `-0.00214395` | `0.000` | `00019` | `-0.00143187` |
| `fscore@0.05` | `-0.00472891` | `0.000` | `00425` | `-0.00258586` |
| `fscore@0.1` | `-0.00673791` | `0.000` | `00425` | `-0.00242105` |

Point-cloud diagnostics:

| protocol | residual mean | residual RMSE | inlier@0.05m |
|---|---:|---:|---:|
| `baseline_camera_anchor_point_residuals` | `0.158708` | `0.204737` | `0.086333` |
| `candidate_camera_anchor_point_residuals` | `0.231392` | `0.308963` | `0.067583` |
| `baseline_old_free_icp_point_residuals` | `0.533749` | `0.649462` | `0.020333` |
| `candidate_old_free_icp_point_residuals` | `0.438592` | `0.540601` | `0.039458` |

## Stage C Deterministic Smoke Conclusion

Stage C regenerated the same smoke candidates as complete v3 caches, removed pre-alignment random 50k reconstruction truncation, and used deterministic hash sampling for metrics.

Runtime and cache policy:

```text
VGGT runtime seconds = 57.941
peak GPU memory bytes = 6158511104
output size bytes = 61806190
max_reconstruction_points = null
reconstruction_sample_method = none
metric_sample_method = hash
max_metric_points = 12000
```

Reconstruction cache counts:

- `23a55d429693f934`: input_shape `[3, 3, 392, 518]`, raw `609168`, filtered `609168`, saved `609168`, sample `none`
- `23a55d429693f934__plus__00010`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00019`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00325`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`
- `23a55d429693f934__plus__00425`: input_shape `[4, 3, 392, 518]`, raw `812224`, filtered `812224`, saved `812224`, sample `none`

Held-out deterministic camera-anchor gains:

| metric | mean gain | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---|---:|
| `chamfer` | `0.41704532` | `1.000` | `00325` | `0.74176456` |
| `accuracy` | `-0.12213880` | `0.000` | `00019` | `-0.00679053` |
| `completeness` | `0.95622945` | `1.000` | `00325` | `1.78152227` |
| `coverage` | `-0.00127778` | `0.333` | `00425` | `0.00050000` |
| `fscore@0.02` | `-0.00234810` | `0.000` | `00425` | `-0.00141536` |
| `fscore@0.05` | `-0.00343737` | `0.000` | `00425` | `-0.00007769` |
| `fscore@0.1` | `-0.00609095` | `0.000` | `00425` | `-0.00138420` |

Point-cloud diagnostics:

| protocol | residual mean | residual RMSE | inlier@0.05m |
|---|---:|---:|---:|
| `baseline_camera_anchor_point_residuals` | `0.156149` | `0.200617` | `0.087667` |
| `candidate_camera_anchor_point_residuals` | `0.247632` | `0.322040` | `0.072021` |
| `baseline_old_free_icp_point_residuals` | `0.526998` | `0.642022` | `0.025750` |
| `candidate_old_free_icp_point_residuals` | `0.443817` | `0.548059` | `0.037042` |

Per-candidate Stage C results:

| candidate | tags | candidate scale | candidate camera RMSE | chamfer gain | accuracy gain | completeness gain | coverage gain | F@0.05 gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `00010` | `duplicate_input_sensitivity` | `1.948739` | `0.012537` | `0.015020` | `0.000485` | `0.029555` | `0.000500` | `0.001003` |
| `00019` | `high_overlap_neighbor` | `1.938811` | `0.013129` | `0.020601` | `-0.006791` | `0.047993` | `-0.000917` | `-0.001769` |
| `00325` | `old_oracle_best` | `2.163518` | `0.015898` | `0.741765` | `-0.297993` | `1.781522` | `-0.003417` | `-0.008465` |
| `00425` | `new_area` | `2.161189` | `0.016536` | `0.488770` | `-0.061633` | `1.039173` | `0.000500` | `-0.000078` |

## Current Interpretation

Removing random 50k pre-alignment truncation did **not** make coverage or F-score reliably positive. The remaining failure is therefore more likely in one of these areas:

- GT frustum overlap and novel visible surface labeling are not yet verified.
- Candidate semantic selection may be wrong; distance alone is not enough to define high-overlap/new-area.
- VGGT candidate reconstructions may introduce outliers that hurt accuracy/F-score even when completeness improves.
- Metric target/visibility protocol may be too global for partial reconstructions.

## Next Step

Do not run full audit20 yet. The next calibration step should compute or inspect GT frustum overlap and novel visible surface fraction for the smoke candidates, especially `00325` and `00425`, then decide whether these are valid new-area cases before expanding reconstruction.
