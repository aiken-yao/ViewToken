# Stage C7 Memory-Anchored Scale Calibration

Status: `complete_with_ranking_stability_caveat`

- `did_run_vggt=false`; reused the six Stage C5 v4 caches from commit `25490b5`.
- No audit20 expansion, new scenes, or policy/probe training.
- 270 replay combinations: B/C/D, memory/per-cache/fixed-baseline scale protocols, per-view quota 3000, target 12k seeds 0..4 and target 50k seed 0.

## Scale calibration

Only the three shared observed views were used. Candidate view, GT novel mask, visibility mask, and full GT point cloud were excluded from fitting. All five candidates calibrated without `blocked_inconsistent_observed_depth_scale`; every observed view had valid ratio fraction 1.0.

| candidate | memory scale | ratio median | ratio MAD | relative median spread |
|---|---:|---:|---:|---:|
| 00018 | 1.9613 | 1.0061 | 0.00108 | 0.00254 |
| 00369 | 1.8864 | 0.9677 | 0.00558 | 0.01072 |
| 00384 | 1.9635 | 1.0073 | 0.00167 | 0.00205 |
| 00065 | 2.0702 | 1.0620 | 0.00227 | 0.00335 |
| 00437 | 1.8762 | 0.9624 | 0.00298 | 0.00314 |

## Replay conclusion

Memory-anchored C/D append-only novel gain is positive at 5 cm and 10 cm for all four non-control candidates across all five 12k seeds; the 50k seed-0 check keeps the positive sign. The `00018` control remains zero at 12k and approximately zero at 50k. Append-only coverage is monotonic by construction and verified by tests.

The `00384` and `00065` history-recompute effects remain negative under memory scale (C/D at 10 cm: approximately -84/-109 and -409/-353 counts at 12k seed 0), so history disturbance remains an independent consequence target.

## Caveat

C7 does not fully pass the ranking-stability requirement. The implementation records all protocols, but the C/D ranking is still sensitive to scale protocol and target size; memory scale removes the need for an after-the-fact per-cache/fixed-baseline choice, but does not establish a universally stable ranking. Therefore only the next-stage 5-scene dataset design is justified; data generation and policy training remain prohibited.

## Verification

- unittest: 76 passed
- compileall: passed
- git diff --check: passed
- official `vggt/`: unchanged
