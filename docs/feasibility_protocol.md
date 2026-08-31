# ViewToken feasibility protocol

## Research question

Can world-aligned VGGT patch features predict the reconstruction improvement
of an unseen candidate view better than geometry and confidence alone?

The project should not proceed to a full NBV policy unless this hypothesis is
supported on held-out scenes.

## No-leakage rule

At decision time, a candidate view is represented only by its camera pose and
rays together with the memory built from already observed images. Candidate
RGB, depth, VGGT features, and ground-truth visibility are forbidden inputs.

Candidate images may be processed offline only to compute oracle labels:

```text
gain(state_t, candidate_v)
  = quality(reconstruct(observed_t + candidate_v), ground_truth)
  - quality(reconstruct(observed_t), ground_truth)
```

For error metrics such as Chamfer distance, reverse the subtraction so that a
larger value always means a better candidate.

## Phase 0: evidence cache

For each observed set, cache:

- raw layer-23 VGGT patch features;
- confidence-weighted 3D position for every patch;
- patch confidence and validity;
- dense depth and world points for geometric baselines;
- camera pose encoding and the exact image/view identifiers.

Do not merge tokens across views in this phase.

## Phase 1: oracle-gain dataset

Start with 10--20 object-centric scenes and roughly 40 candidate cameras per
scene. Use 3--4 initial observations. For every held-out candidate, reconstruct
with and without that view and store:

- Chamfer-distance reduction;
- F-score improvement at fixed thresholds;
- surface-coverage improvement;
- current view IDs and candidate pose;
- scene split and reconstruction alignment metadata.

Use the same Sim(3) alignment protocol for every method. Split by scene/object,
never by individual view pair.

## Phase 2: diagnostic probes

Train equal-capacity gain predictors with the following inputs:

1. candidate pose only;
2. observed geometry and visibility;
3. observed xyz plus confidence (AREA3D-style geometric baseline);
4. VGGT feature plus xyz, confidence, and candidate pose/rays.

Report Spearman and Kendall rank correlation, Top-1/Top-3 accuracy, and Top-1
regret:

```text
regret = oracle_best_gain - predicted_choice_gain
```

## Go/no-go criterion

Proceed to the counterfactual ViewToken policy only if the token probe improves
ranking and regret consistently on unseen scenes relative to xyz plus
confidence. A useful project-level target is at least a 5% relative reduction
in Top-1 regret, accompanied by consistent per-scene results rather than an
average dominated by a few easy scenes.

If token features do not improve the geometric baseline, stop and diagnose
cross-view alignment before adding model complexity.

