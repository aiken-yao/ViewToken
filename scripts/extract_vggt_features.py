"""Extract VGGT patch tokens and geometry for a set of observed images."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt"
for import_root in (PROJECT_ROOT, VGGT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from viewtoken.backbones import VGGTFeatureExtractor  # noqa: E402
from viewtoken.memory import build_patch_scene_tokens  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "extract_vggt.yaml",
    )
    parser.add_argument("--images", type=Path, nargs="+", default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}, received {type(config).__name__}")
    return config


def resolve_images(args: argparse.Namespace, config: dict[str, Any]) -> list[Path]:
    if args.images:
        image_paths = args.images
    else:
        configured_images = config.get("images") or []
        image_paths = [Path(path) for path in configured_images]

    image_dir_value = args.image_dir or config.get("image_dir")
    if image_dir_value:
        image_dir = Path(image_dir_value)
        image_paths.extend(
            path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    unique_paths = list(dict.fromkeys(path.expanduser().resolve() for path in image_paths))
    if not unique_paths:
        raise ValueError("No input images were provided")

    missing = [path for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input images: " + ", ".join(map(str, missing)))
    return unique_paths


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_compute_dtype(config: dict[str, Any], device: torch.device) -> torch.dtype:
    requested = str(config.get("compute_dtype", "auto"))
    if requested != "auto":
        try:
            return TORCH_DTYPES[requested]
        except KeyError as error:
            raise ValueError(f"Unsupported compute dtype: {requested}") from error

    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def cast_for_storage(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    try:
        dtype = TORCH_DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported storage dtype: {dtype_name}") from error
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_paths = resolve_images(args, config)
    output_dir = (args.output_dir or Path(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config)
    compute_dtype = resolve_compute_dtype(config, device)
    model_id = str(config.get("model_id", "facebook/VGGT-1B"))
    preprocess_mode = str(config.get("preprocess_mode", "crop"))
    layer_index = int(config.get("layer_index", 23))

    print(f"Loading {model_id} on {device}...")
    model = VGGT.from_pretrained(model_id).to(device)
    model.eval().requires_grad_(False)

    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=preprocess_mode
    ).to(device)
    extractor = VGGTFeatureExtractor(model, layer_index=layer_index)

    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        features = extractor.extract(images)

    scene_tokens = build_patch_scene_tokens(
        patch_tokens=features.patch_tokens,
        world_points=features.world_points,
        world_points_conf=features.world_points_conf,
        patch_grid=features.patch_grid,
        min_confidence=float(config.get("min_world_point_confidence", 0.0)),
    )

    token_dtype = str(config.get("token_dtype", "float16"))
    geometry_dtype = str(config.get("geometry_dtype", "float32"))
    stored_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in features.tensor_dict().items():
        storage_dtype = token_dtype if name == "patch_tokens" else geometry_dtype
        stored = cast_for_storage(tensor, storage_dtype)
        torch.save(stored, output_dir / f"{name}.pt")
        stored_tensors[name] = stored

    scene_tensor_specs = {
        "patch_positions": (scene_tokens.positions, geometry_dtype),
        "patch_confidence": (scene_tokens.confidence, geometry_dtype),
        "patch_valid_mask": (scene_tokens.valid_mask, None),
    }
    for name, (tensor, dtype_name) in scene_tensor_specs.items():
        stored = (
            tensor.detach().to(device="cpu").contiguous()
            if dtype_name is None
            else cast_for_storage(tensor, dtype_name)
        )
        torch.save(stored, output_dir / f"{name}.pt")
        stored_tensors[name] = stored

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "image_paths": [str(path) for path in image_paths],
        "preprocess_mode": preprocess_mode,
        "input_shape": list(images.shape),
        "layer_index": features.layer_index,
        "patch_start_idx": features.patch_start_idx,
        "patch_grid": list(features.patch_grid),
        "min_world_point_confidence": float(
            config.get("min_world_point_confidence", 0.0)
        ),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "tensor_shapes": {
            name: list(tensor.shape) for name, tensor in stored_tensors.items()
        },
        "tensor_dtypes": {
            name: str(tensor.dtype).removeprefix("torch.")
            for name, tensor in stored_tensors.items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved VGGT evidence for {len(image_paths)} images to {output_dir}")
    for name, tensor in stored_tensors.items():
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
