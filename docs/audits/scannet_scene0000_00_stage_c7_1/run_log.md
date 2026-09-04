# Stage C7.1 Corrected Frozen-Baseline Scale Audit

Status: `passed`

## Corrections

The original Stage C7 report is preserved as historical evidence, but its memory-protocol H0 metrics and claimed target-seed variation are invalid. C7.1 fixes both issues:

- H0 always uses fixed `base_scale=1.9494023` and one shared 9000-point tensor per branch.
- Only candidate H1/H2/H3 geometry uses the selected candidate protocol scale.
- H2 preserves H0 as an exact element-wise prefix.
- Target sampling uses real seeds 0..4; reconstruction per-view sampling is separately fixed at seed 0.
- B has only per-cache and fixed-baseline protocols. Memory scale is defined only for C/D.

## Execution

- Reused six v4 caches from commit `25490b5`.
- `did_run_vggt=false`; official `vggt/` unchanged.
- Per-view quota: 3000.
- 12k target: five actual hash samples, seeds 0..4, from 50k source points.
- 50k target: seed 0, no target subsampling.
- Total records: 240.

## Memory-Protocol Results

All four non-control candidates have positive C/D append-only novel gain at 5 cm and 10 cm for every 12k target seed and for the 50k check. The `00018` control remains zero.

| Branch | Candidate | Gain @5 cm mean +/- std | Gain @10 cm mean +/- std | Positive fraction @10 cm | History effect @10 cm |
|---|---|---:|---:|---:|---:|
| C | 00369 | 47.4 +/- 3.6 | 220.8 +/- 8.5 | 1.0 | -37.8 |
| C | 00384 | 152.6 +/- 3.8 | 331.4 +/- 14.1 | 1.0 | -14.8 |
| C | 00065 | 108.4 +/- 2.7 | 296.2 +/- 8.5 | 1.0 | +20.0 |
| C | 00437 | 119.2 +/- 9.9 | 231.2 +/- 18.1 | 1.0 | -2.8 |
| D | 00369 | 45.8 +/- 5.0 | 230.0 +/- 10.7 | 1.0 | -48.2 |
| D | 00384 | 150.6 +/- 3.0 | 325.4 +/- 16.0 | 1.0 | -26.8 |
| D | 00065 | 89.0 +/- 4.9 | 296.8 +/- 12.2 | 1.0 | +19.6 |
| D | 00437 | 121.2 +/- 8.1 | 243.2 +/- 17.5 | 1.0 | +3.8 |

The corrected result changes the interpretation of `00065`: its memory-scale history effect is slightly positive, so the original C7 claim of negative history disturbance for `00065` is invalid. `00384` remains consistently negative and should remain an independent consequence target.

## Ranking And Uncertainty

C and D produce the same order for every 12k target seed and for the 50k target:

`00384 > 00065 > 00437 ~= 00369 > 00018`

- 12k aggregate versus 50k Spearman: 1.0 for C and D.
- C versus D Spearman: 1.0.
- `00369` and `00437` have overlapping one-standard-deviation intervals at 10 cm and are recorded as an uncertainty tie, not a strict pairwise label.

## Verification

- unittest: 79 passed
- compileall: passed
- git diff --check: passed
- new VGGT forwards: 0
- audit20 expansion, multi-scene generation, probe/policy training: none

C7.1 passes. The next step may design a five-scene small oracle dataset and uncertainty-aware supervision, but this run does not generate that dataset or train a model.
