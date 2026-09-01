# Oracle Gain Audit Log: ScanNet `scene0000_00`

Date: 2026-09-01

This log records the current development-stage oracle-gain audit for collaborators. It intentionally includes lightweight records and summaries only. Model weights, ScanNet RGB/GT data, dense reconstruction tensors, token caches, core dumps, and full `outputs/` directories are not committed.

## Source State

- Repository: `https://github.com/aiken-yao/ViewToken.git`
- Branch: `main`
- VGGT checkpoint used locally: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/model_weights/vggt/model.pt`
- Working Python used locally: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python`
- Known bad environment: the older torch `2.3.1` Conda environment triggered a 4-view H20 `SIGFPE` in VGGT camera-head MLP/Linear. It was not debugged further for this stage.

## Phase-0 Smoke Result

- Input shape: `[4, 3, 392, 518]`
- `patch_tokens`: `[1, 4, 1036, 2048]`
- `patch_grid`: `[28, 37]`
- `patch_start_idx`: `5`
- `aggregator_forward_count`: `1`
- `patch_valid_ratio`: `1.0`
- patch confidence min/median/max: `1.0573471784591675` / `3.724500894546509` / `9.158475875854492`
- runtime seconds: `13.432809881865978`
- peak GPU memory bytes: `6157936128`
- cache size bytes: `36550473`

## Oracle Audit Inputs

- Scene: `scene0000_00`
- Observed views: `00000`, `00010`, `00020`
- Candidate records: `21`
- Held-out candidate records: `20`
- Candidate image rule: candidate RGB is used only offline for reconstruction-gain labels and is not an NBV policy input.
- Candidate append rule: `True`
- Alignment: `sim3_icp`
- Random seed: `0`
- Confidence threshold: `0.0`
- Voxel downsample size after alignment: `0.02 m`
- Metric point sample count: `12000`
- F-score thresholds: `[0.02, 0.05, 0.1] m`
- Coverage radius: `0.05 m`
- Metric units: meters after Sim(3) alignment to ScanNet GT.

## Held-Out Gain Distribution

All gains use a larger-is-better convention. Chamfer, accuracy, and completeness are distance reductions; coverage and F-score are metric increases.

| metric | min | median | mean | max | positive ratio | oracle-best | oracle-best gain |
|---|---:|---:|---:|---:|---:|---|---:|
| `chamfer` | `-0.340112` | `-0.132436` | `-0.125187` | `0.040855` | `0.05` | `00009` | `0.040855` |
| `accuracy` | `-0.032404` | `0.122480` | `0.129974` | `0.300967` | `0.80` | `00325` | `0.300967` |
| `completeness` | `-0.914911` | `-0.384647` | `-0.380348` | `0.114114` | `0.05` | `00009` | `0.114114` |
| `coverage` | `-0.009833` | `-0.002042` | `-0.001900` | `0.006083` | `0.40` | `00325` | `0.006083` |
| `fscore@0.02` | `-0.000727` | `0.000174` | `0.000420` | `0.002990` | `0.65` | `00325` | `0.002990` |
| `fscore@0.05` | `-0.008741` | `0.001225` | `0.000883` | `0.016139` | `0.60` | `00325` | `0.016139` |
| `fscore@0.1` | `-0.042125` | `-0.012260` | `-0.009991` | `0.019378` | `0.35` | `00325` | `0.019378` |

## Spearman Rank Correlation

| metric pair | rho |
|---|---:|
| `chamfer|accuracy` | `-0.445113` |
| `chamfer|completeness` | `0.917293` |
| `chamfer|coverage` | `0.725837` |
| `accuracy|completeness` | `-0.706767` |
| `completeness|coverage` | `0.726589` |
| `coverage|fscore@0.05` | `0.903347` |
| `coverage|fscore@0.1` | `0.951486` |

## Sanity Checks

### repeat_observed
Expected: Gain should be close to zero because the view is already in memory.

| candidate | pose distance to observed | chamfer | accuracy | completeness | coverage | F@0.02 | F@0.05 | F@0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `00010` | `0.000000 m` | `0.049566` | `-0.025109` | `0.124241` | `0.005750` | `-0.000033` | `0.004993` | `0.013572` |

### high_overlap_neighbor
Expected: Gain is expected to be small for views with high overlap to observed inputs.

| candidate | pose distance to observed | chamfer | accuracy | completeness | coverage | F@0.02 | F@0.05 | F@0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `00019` | `0.008193 m` | `-0.015325` | `-0.002396` | `-0.028253` | `0.003250` | `0.000217` | `0.003324` | `0.004827` |
| `00018` | `0.008384 m` | `-0.009301` | `-0.002684` | `-0.015918` | `0.002333` | `0.000622` | `0.003062` | `0.007242` |
| `00009` | `0.035085 m` | `0.040855` | `-0.032404` | `0.114114` | `0.002667` | `-0.000449` | `0.001751` | `0.008731` |
| `00032` | `0.051909 m` | `-0.109010` | `0.077968` | `-0.295988` | `-0.005083` | `-0.000699` | `-0.004214` | `-0.016893` |

### new_area
Expected: At least one coverage or completeness-oriented metric should improve if new surfaces are observed.

| candidate | pose distance to observed | chamfer | accuracy | completeness | coverage | F@0.02 | F@0.05 | F@0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `00425` | `3.914814 m` | `-0.134458` | `0.107164` | `-0.376080` | `-0.002583` | `-0.000133` | `0.001492` | `-0.019348` |
| `00500` | `5.026197 m` | `-0.275119` | `0.233758` | `-0.783996` | `-0.003417` | `0.000872` | `0.001928` | `-0.016811` |

## Current Interpretation

The audit is runnable and produces a non-degenerate distribution, but it is not clean enough to start scaling the oracle-gain dataset yet.

- Chamfer and completeness have mostly negative held-out gains: chamfer positive ratio `0.05`, completeness positive ratio `0.05`.
- Accuracy improves for many candidates, but this conflicts with completeness/coverage, suggesting added views may reduce prediction-to-GT error while losing or misaligning GT-to-prediction coverage.
- Repeat-observed `00010` is not close to all-zero gain, so deterministic reconstruction, point sampling, alignment, or duplicate-view handling should be audited before treating these labels as final.
- New-area probes `00425` and `00500` improve accuracy but do not improve completeness/coverage under the current protocol.

## Commands Run

```bash
# Phase-0 extraction smoke
CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 \
  /mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python -u \
  scripts/extract_vggt_features.py \
  --checkpoint /mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/model_weights/vggt/model.pt \
  --images \
    /mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00/00000.jpg \
    /mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00/00010.jpg \
    /mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00/00020.jpg \
    /mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00/00030.jpg \
  --output-dir outputs/token_cache/smoke_test_scannet_scene0000_00

# Oracle audit, 21 candidates including 20 held-out
CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 \
  /mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python -u \
  scripts/generate_oracle_gain.py \
  --config configs/oracle_gain_scannet_audit20.yaml

# Tests
/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python -m compileall -q viewtoken scripts tests
/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/vggt/.venv/vggt-nv-sys/bin/python -m unittest discover -s tests -v
```

## Validation

- `compileall`: passed
- `unittest discover -s tests -v`: 16 tests passed
- Oracle JSONL parse check: passed, 21 records
- All cached reconstruction metadata reported `aggregator_forward_count = 1`
- Audit output size copied here is lightweight: `audit_summary.json`, `metadata.json`, and `oracle_gain.jsonl`

## Recommended Next Checks

1. Make repeat-observed reconstruction deterministic or explicitly exclude repeat-observed from dataset records after using it as a sanity check.
2. Compare `identity` alignment vs Sim(3) ICP on the same cached points to isolate alignment effects.
3. Re-run with stricter world-point confidence thresholds and inspect coverage/completeness sensitivity.
4. Verify whether far-view candidates introduce outliers that improve accuracy while hurting completeness.
5. Do not start full policy training until these sanity checks are stable.
