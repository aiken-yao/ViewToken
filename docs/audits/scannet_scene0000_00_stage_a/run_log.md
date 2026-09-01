# Stage A Oracle Metric Stability Audit

This report uses cached reconstruction points only. It does not run VGGT or create new reconstructions.

## Inputs

- Source points: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934/points.pt`
- Target points: `/mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/points/scene0000_00.bin`
- Alignment: `sim3_icp`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Voxel downsample size: `0.02`
- Max metric points: `12000`
- Trim fraction: `1.0`
- Inlier threshold: `0.05`

## Identical-Cloud Check

- Max absolute gain: `0`
- Gains: `{"accuracy": 0.0, "chamfer": 0.0, "completeness": 0.0, "coverage": 0.0, "fscore@0.02": 0.0, "fscore@0.05": 0.0, "fscore@0.1": 0.0}`

## Multi-Seed Metric Stability

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| `accuracy` | `0.53635204` | `0.00284753` | `0.53161347` | `0.54189223` |
| `chamfer` | `0.68655644` | `0.00795060` | `0.67333701` | `0.69790313` |
| `completeness` | `0.83676085` | `0.01813601` | `0.80478179` | `0.86419278` |
| `coverage` | `0.01658333` | `0.00151153` | `0.01466667` | `0.01975000` |
| `fscore@0.02` | `0.00191105` | `0.00044062` | `0.00141667` | `0.00266667` |
| `fscore@0.05` | `0.01915900` | `0.00156658` | `0.01718588` | `0.02187006` |
| `fscore@0.1` | `0.07201726` | `0.00212163` | `0.06856164` | `0.07603556` |

## Multi-Seed Alignment Diagnostics

| diagnostic | mean | std | min | max |
|---|---:|---:|---:|---:|
| `scale` | `5.12071822` | `0.03825520` | `5.04755988` | `5.18516153` |
| `rotation_angle_degrees` | `11.89984503` | `0.64862965` | `11.09473006` | `12.89692649` |
| `translation_norm` | `8.44467897` | `0.04135654` | `8.34442425` | `8.48712444` |
| `residual_mean` | `0.56083169` | `0.00722667` | `0.54874146` | `0.57327724` |
| `residual_median` | `0.49508245` | `0.00961100` | `0.47661453` | `0.50935572` |
| `residual_rmse` | `0.66883979` | `0.00709923` | `0.65495354` | `0.68128449` |
| `residual_max` | `1.80038912` | `0.03014410` | `1.76203382` | `1.86453485` |
| `inlier_ratio` | `0.00869141` | `0.00151367` | `0.00659180` | `0.01171875` |

## Existing Candidate Gain Spread

Existing records: `21`, held-out: `20`.

| metric | metric std across seeds | candidate gain range | ratio |
|---|---:|---:|---:|
| `accuracy` | `0.00284753` | `0.33337045` | `0.00854165` |
| `chamfer` | `0.00795060` | `0.38096699` | `0.02086953` |
| `completeness` | `0.01813601` | `1.02902520` | `0.01762446` |
| `coverage` | `0.00151153` | `0.01591667` | `0.09496524` |
| `fscore@0.02` | `0.00044062` | `0.00371743` | `0.11852727` |
| `fscore@0.05` | `0.00156658` | `0.02488028` | `0.06296490` |
| `fscore@0.1` | `0.00212163` | `0.06150293` | `0.03449636` |

## Interpretation

The identical-cloud check should be zero up to floating-point noise. Multi-seed standard deviations should be much smaller than real candidate gain differences before the oracle labels can be trusted.

Runtime seconds: `9.394`
