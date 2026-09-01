"""Extract VGGT patch tokens and geometry for a set of observed images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local VGGT training checkpoint containing a model state dict.",
    )
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


def resolve_checkpoint_path(
    args: argparse.Namespace, config: dict[str, Any]
) -> Path | None:
    checkpoint_value = args.checkpoint or config.get("checkpoint_path")
    if not checkpoint_value:
        return None

    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return checkpoint_path


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


def _strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix(prefix): tensor for key, tensor in state_dict.items()}


def extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate

        if checkpoint and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in checkpoint.items()
        ):
            return checkpoint

    raise TypeError(
        "Checkpoint must be a state dict or contain one under "
        "'model', 'state_dict', or 'model_state_dict'"
    )


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    load_kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    try:
        checkpoint = torch.load(path, mmap=True, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(path, **load_kwargs)

    state_dict = extract_model_state_dict(checkpoint)
    state_dict = _strip_state_dict_prefix(state_dict, "module.")
    return _strip_state_dict_prefix(state_dict, "model.")


def checkpoint_has_module(state_dict: Mapping[str, torch.Tensor], module_name: str) -> bool:
    return any(key.startswith(f"{module_name}.") for key in state_dict)


def load_vggt_model(model_id: str, checkpoint_path: Path | None) -> VGGT:
    if checkpoint_path is None:
        return VGGT.from_pretrained(model_id)

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    required_modules = ("aggregator", "camera_head", "depth_head", "point_head")
    missing_required = [
        module_name
        for module_name in required_modules
        if not checkpoint_has_module(state_dict, module_name)
    ]
    if missing_required:
        raise RuntimeError(
            "Local checkpoint is missing VGGT modules required for Phase-0: "
            + ", ".join(missing_required)
        )

    model = VGGT(enable_track=checkpoint_has_module(state_dict, "track_head"))
    model.load_state_dict(state_dict, strict=True)
    return model


def tensor_finite_ratio(tensor: torch.Tensor) -> float | None:
    if not tensor.is_floating_point():
        return None
    return torch.isfinite(tensor).float().mean().item()


def confidence_stats(confidence: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    valid_confidence = confidence.detach()[valid_mask.detach()]
    if valid_confidence.numel() == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    valid_confidence = valid_confidence.float().cpu()
    return {
        "min": valid_confidence.min().item(),
        "median": valid_confidence.median().item(),
        "max": valid_confidence.max().item(),
    }


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    image_paths = resolve_images(args, config)
    output_dir = (args.output_dir or Path(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config)
    compute_dtype = resolve_compute_dtype(config, device)
    model_id = str(config.get("model_id", "facebook/VGGT-1B"))
    checkpoint_path = resolve_checkpoint_path(args, config)
    preprocess_mode = str(config.get("preprocess_mode", "crop"))
    layer_index = int(config.get("layer_index", 23))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if checkpoint_path is None:
        print(f"Loading {model_id} on {device}...")
    else:
        print(f"Loading VGGT from checkpoint {checkpoint_path} on {device}...")
    model = load_vggt_model(model_id, checkpoint_path).to(device)
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
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "image_paths": [str(path) for path in image_paths],
        "preprocess_mode": preprocess_mode,
        "input_shape": list(images.shape),
        "layer_index": features.layer_index,
        "patch_start_idx": features.patch_start_idx,
        "aggregator_forward_count": features.aggregator_forward_count,
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
    metadata["tensor_finite_ratios"] = {
        name: tensor_finite_ratio(tensor) for name, tensor in stored_tensors.items()
    }
    metadata["patch_valid_ratio"] = scene_tokens.valid_mask.float().mean().item()
    metadata["patch_confidence_stats"] = confidence_stats(
        scene_tokens.confidence, scene_tokens.valid_mask
    )
    metadata["runtime_seconds"] = time.perf_counter() - start_time
    metadata["peak_gpu_memory_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    metadata["cache_size_bytes"] = directory_size_bytes(output_dir)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved VGGT evidence for {len(image_paths)} images to {output_dir}")
    for name, tensor in stored_tensors.items():
        finite_ratio = metadata["tensor_finite_ratios"][name]
        print(
            f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"finite_ratio={finite_ratio}"
        )
    print(f"  patch_valid_ratio: {metadata['patch_valid_ratio']:.6f}")
    print(f"  patch_confidence: {metadata['patch_confidence_stats']}")
    print(f"  aggregator_forward_count: {features.aggregator_forward_count}")
    print(f"  runtime_seconds: {metadata['runtime_seconds']:.3f}")
    print(f"  peak_gpu_memory_bytes: {metadata['peak_gpu_memory_bytes']}")
    print(f"  cache_size_bytes: {metadata['cache_size_bytes']}")


if __name__ == "__main__":
    main()
