# ViewToken

ViewToken studies token-level next-best-view planning for active 3D
reconstruction. The core hypothesis is that world-aligned VGGT patch features
contain view-planning utility beyond geometry and confidence alone.

The repository currently contains the official VGGT project under `vggt/` and
a Phase-0 feature extraction pipeline under `viewtoken/`.

The experimental go/no-go protocol is documented in
[`docs/feasibility_protocol.md`](docs/feasibility_protocol.md).
Instructions for the GPU-side Codex agent are in
[`REMOTE_CODEX_TIPS.md`](REMOTE_CODEX_TIPS.md).

## Phase 0: extract VGGT scene evidence

Install a CUDA-compatible PyTorch build first, then install the local
dependencies:

```bash
pip install -r requirements.txt
```

Edit `configs/extract_vggt.yaml`, or pass image paths directly:

```bash
python scripts/extract_vggt_features.py \
  --images path/to/view_000.png path/to/view_001.png \
  --output-dir outputs/token_cache/example
```

The extractor performs one frozen VGGT forward pass and writes:

- `patch_tokens.pt`: raw layer-23 patch tokens after removing special tokens;
- `depth.pt` and `depth_conf.pt`;
- `world_points.pt` and `world_points_conf.pt`;
- `patch_positions.pt`, `patch_confidence.pt`, and `patch_valid_mask.pt`,
  which align each patch token with a confidence-weighted 3D position;
- `pose_enc.pt` when the camera head is enabled;
- `metadata.json` with image paths, tensor shapes, patch grid, and dtypes.

Candidate-view RGB or depth must never be used by the future NBV policy at
inference time. Held-out candidate images are only permitted when constructing
oracle reconstruction-gain labels during training and evaluation.

Run the dependency-free test harness (after installing PyTorch):

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
ViewToken/
  configs/                 Experiment configuration
  scripts/                 Command-line entry points
  tests/                   Lightweight unit tests
  vggt/                    Official VGGT source tree
  viewtoken/               ViewToken implementation
```
