# Stage C6 reconstruction disturbance audit

Status: `complete_with_scale_calibration_blocker`

## Protocol

- Reused the six Stage C5 v4 caches from commit `25490b5`.
- `did_run_vggt=false`; official `vggt/` was not modified.
- Branches: B/C/D.
- States: H0 baseline, H1 observed-recomputed-only, H2 append-only, H3 joint-recompute.
- Per-view deterministic hash quota: 3000 points.
- Target sweeps: 12k with seeds 0..4, then 50k with seed 0.
- Scale protocols: per-cache observed-anchor scale and fixed baseline scale 1.9494023.
- Total evaluated combinations: 180.

## Findings

Append-only candidate addition is stable: all four non-control candidates have positive novel covered-count effects at 5 cm and 10 cm for B/C/D, under both scale protocols, across all five 12k seeds. The 50k seed-0 check preserves the positive sign. The `00018` near-duplicate control remains approximately zero.

Append-only observed coverage never decreases, as required by set monotonicity. The large C5 retention losses for `00384` and `00065` appear only after replacing the frozen H0 history with jointly recomputed observed geometry:

| branch/protocol | 00384 history effect @10cm | 00065 history effect @10cm |
|---|---:|---:|
| C per-cache | -161.8 | -125.6 |
| D per-cache | -185.4 | -139.8 |
| C fixed baseline | -98.2 | -406.8 |
| D fixed baseline | -107.8 | -451.4 |

This supports a frozen Scene Token Memory plus append-only fusion protocol. Joint recomputation introduces a real history disturbance for `00384` and `00065`; it is not caused by candidate points displacing observed points under a global 12k sample.

## Scale blocker

Candidate ranking is not sufficiently stable across scale protocols:

| Branch | Spearman |
|---|---:|
| B | 1.0 |
| C | 0.4 |
| D | 0.7 |

For `00369` and `00437`, the observed history-recompute effect can also flip sign between per-cache and fixed-baseline scale. Therefore C6 does not yet approve automatic multi-scene dataset expansion. Scale calibration must be resolved before assigning stable ranking labels.

## Verification

- unittest: 71 passed
- compileall: passed
- git diff --check: passed
- new VGGT forwards: 0
- audit20 expansion: none
- policy training: none
