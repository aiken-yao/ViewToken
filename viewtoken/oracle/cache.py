"""Cache fingerprints and integrity checks for oracle reconstruction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = "oracle-reconstruction-v3"
REQUIRED_RECONSTRUCTION_ARTIFACTS = (
    "points.pt",
    "confidence.pt",
    "pose_enc.pt",
    "metadata.json",
)
CACHE_SCHEMA_VERSION_V4 = "oracle-reconstruction-v4"
REQUIRED_RECONSTRUCTION_ARTIFACTS_V4 = (
    *REQUIRED_RECONSTRUCTION_ARTIFACTS,
    "depth.pt",
    "depth_conf.pt",
    "predicted_intrinsics.pt",
    "transformed_gt_intrinsics.pt",
)


class ReconstructionCacheValidationError(RuntimeError):
    """Raised when an oracle reconstruction cache is incomplete or stale."""

    def __init__(self, reconstruction_dir: Path, errors: list[str]) -> None:
        self.reconstruction_dir = reconstruction_dir
        self.errors = list(errors)
        joined = "; ".join(errors)
        super().__init__(f"Invalid reconstruction cache {reconstruction_dir}: {joined}")


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_reconstruction_cache_payload(
    checkpoint_path: Path,
    image_paths: list[Path],
    preprocess_mode: str,
    layer_index: int,
    min_confidence: float,
    max_points: int | None,
    seed: int,
    sample_method: str = "random",
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "checkpoint": file_identity(checkpoint_path),
        "ordered_images": [file_identity(path) for path in image_paths],
        "preprocess_mode": preprocess_mode,
        "layer_index": layer_index,
        "min_world_point_confidence": min_confidence,
        "max_reconstruction_points": max_points,
        "reconstruction_sample_method": sample_method,
        "seed": seed,
    }


def hash_cache_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_reconstruction_cache_identity(
    checkpoint_path: Path,
    image_paths: list[Path],
    preprocess_mode: str,
    layer_index: int,
    min_confidence: float,
    max_points: int | None,
    seed: int,
    sample_method: str = "random",
) -> dict[str, Any]:
    payload = build_reconstruction_cache_payload(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        preprocess_mode=preprocess_mode,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_points,
        seed=seed,
        sample_method=sample_method,
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": hash_cache_payload(payload),
        "payload": payload,
    }


def build_v4_reconstruction_cache_payload(
    checkpoint_path: Path,
    image_paths: list[Path],
    preprocess_mode: str,
    layer_index: int,
    min_confidence: float,
    max_points: int | None,
    seed: int,
    sample_method: str = "none",
    preprocessing_transforms: list[dict[str, Any]] | None = None,
    per_view_shape_offsets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_reconstruction_cache_payload(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        preprocess_mode=preprocess_mode,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_points,
        seed=seed,
        sample_method=sample_method,
        schema_version=CACHE_SCHEMA_VERSION_V4,
    )
    payload["preprocessing_transforms"] = preprocessing_transforms or []
    payload["per_view_shape_offsets"] = per_view_shape_offsets or []
    payload["required_artifacts"] = list(REQUIRED_RECONSTRUCTION_ARTIFACTS_V4)
    payload["depth_backprojection_schema"] = {
        "depth_units": "VGGT predicted camera Z before any known-pose scale calibration",
        "intrinsics_branches": [
            "predicted_intrinsics",
            "transformed_calibrated_intrinsics",
        ],
    }
    return payload


def build_v4_reconstruction_cache_identity(
    checkpoint_path: Path,
    image_paths: list[Path],
    preprocess_mode: str,
    layer_index: int,
    min_confidence: float,
    max_points: int | None,
    seed: int,
    sample_method: str = "none",
    preprocessing_transforms: list[dict[str, Any]] | None = None,
    per_view_shape_offsets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_v4_reconstruction_cache_payload(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        preprocess_mode=preprocess_mode,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_points,
        seed=seed,
        sample_method=sample_method,
        preprocessing_transforms=preprocessing_transforms,
        per_view_shape_offsets=per_view_shape_offsets,
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION_V4,
        "fingerprint": hash_cache_payload(payload),
        "payload": payload,
    }


def _load_torch_artifact_shape(path: Path) -> tuple[int, ...]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not hasattr(artifact, "shape"):
        raise TypeError(f"{path.name} must contain a tensor-like object")
    return tuple(int(value) for value in artifact.shape)


def _normalize_depth_shape(shape: tuple[int, ...], label: str) -> tuple[int, int, int]:
    if len(shape) == 3:
        return shape
    if len(shape) == 4 and shape[-1] == 1:
        return shape[:3]
    if len(shape) == 4 and shape[0] == 1:
        return shape[1:]
    raise ValueError(
        f"{label} must have shape [S, H, W], [S, H, W, 1], or [1, S, H, W], got {shape}"
    )


def _normalize_intrinsics_shape(shape: tuple[int, ...], label: str) -> tuple[int, int, int]:
    if len(shape) == 3 and shape[-2:] == (3, 3):
        return shape
    if len(shape) == 4 and shape[0] == 1 and shape[-2:] == (3, 3):
        return shape[1:]
    raise ValueError(f"{label} must have shape [S, 3, 3] or [1, S, 3, 3], got {shape}")


def _validate_v4_metadata_layout(
    reconstruction_dir: Path,
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    input_shape = metadata.get("input_shape")
    if isinstance(input_shape, list | tuple) and len(input_shape) == 4:
        view_count = int(input_shape[0])
        input_height = int(input_shape[2])
        input_width = int(input_shape[3])
    else:
        view_count = None
        input_height = None
        input_width = None
        errors.append(f"input_shape must be [S, C, H, W], got {input_shape}")

    transforms = metadata.get("preprocessing_transforms")
    if not isinstance(transforms, list):
        errors.append("preprocessing_transforms must be a list")
    elif view_count is not None and len(transforms) != view_count:
        errors.append(
            "preprocessing_transforms length mismatch: "
            f"expected {view_count}, got {len(transforms)}"
        )

    offsets = metadata.get("per_view_shape_offsets")
    total_count = 0
    if not isinstance(offsets, list):
        errors.append("per_view_shape_offsets must be a list")
    else:
        if view_count is not None and len(offsets) != view_count:
            errors.append(
                "per_view_shape_offsets length mismatch: "
                f"expected {view_count}, got {len(offsets)}"
            )
        expected_offset = 0
        for index, row in enumerate(offsets):
            if not isinstance(row, dict):
                errors.append(f"per_view_shape_offsets[{index}] must be an object")
                continue
            try:
                height = int(row["height"])
                width = int(row["width"])
                offset = int(row.get("point_offset", row.get("offset")))
                count = int(row["point_count"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"per_view_shape_offsets[{index}] is invalid: {exc}")
                continue
            if offset != expected_offset:
                errors.append(
                    f"per_view_shape_offsets[{index}] point_offset mismatch: "
                    f"expected {expected_offset}, got {offset}"
                )
            if count != height * width:
                errors.append(
                    f"per_view_shape_offsets[{index}] point_count must equal height*width"
                )
            if input_height is not None and height != input_height:
                errors.append(
                    f"per_view_shape_offsets[{index}] height mismatch: "
                    f"expected {input_height}, got {height}"
                )
            if input_width is not None and width != input_width:
                errors.append(
                    f"per_view_shape_offsets[{index}] width mismatch: "
                    f"expected {input_width}, got {width}"
                )
            expected_offset += count
        total_count = expected_offset

    try:
        depth_shape = _normalize_depth_shape(
            _load_torch_artifact_shape(reconstruction_dir / "depth.pt"),
            "depth.pt",
        )
    except Exception as exc:  # noqa: BLE001 - validation collects all artifact errors.
        errors.append(f"depth.pt shape validation failed: {exc}")
        depth_shape = None
    try:
        depth_conf_shape = _normalize_depth_shape(
            _load_torch_artifact_shape(reconstruction_dir / "depth_conf.pt"),
            "depth_conf.pt",
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"depth_conf.pt shape validation failed: {exc}")
        depth_conf_shape = None
    try:
        predicted_intrinsics_shape = _normalize_intrinsics_shape(
            _load_torch_artifact_shape(reconstruction_dir / "predicted_intrinsics.pt"),
            "predicted_intrinsics.pt",
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"predicted_intrinsics.pt shape validation failed: {exc}")
        predicted_intrinsics_shape = None
    try:
        transformed_gt_intrinsics_shape = _normalize_intrinsics_shape(
            _load_torch_artifact_shape(reconstruction_dir / "transformed_gt_intrinsics.pt"),
            "transformed_gt_intrinsics.pt",
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"transformed_gt_intrinsics.pt shape validation failed: {exc}")
        transformed_gt_intrinsics_shape = None

    for label, shape in (
        ("depth_conf.pt", depth_conf_shape),
        ("predicted_intrinsics.pt", predicted_intrinsics_shape),
        ("transformed_gt_intrinsics.pt", transformed_gt_intrinsics_shape),
    ):
        if shape is not None and depth_shape is not None and int(shape[0]) != int(depth_shape[0]):
            errors.append(f"{label} view count mismatch: expected {depth_shape[0]}, got {shape[0]}")
    if depth_shape is not None:
        depth_views, depth_height, depth_width = depth_shape
        if view_count is not None and depth_views != view_count:
            errors.append(f"depth.pt view count mismatch: expected {view_count}, got {depth_views}")
        if input_height is not None and depth_height != input_height:
            errors.append(f"depth.pt height mismatch: expected {input_height}, got {depth_height}")
        if input_width is not None and depth_width != input_width:
            errors.append(f"depth.pt width mismatch: expected {input_width}, got {depth_width}")
        if total_count and total_count != depth_views * depth_height * depth_width:
            errors.append(
                "per_view_shape_offsets total point_count mismatch: "
                f"expected {depth_views * depth_height * depth_width}, got {total_count}"
            )
    if depth_shape is not None and depth_conf_shape is not None and depth_conf_shape != depth_shape:
        errors.append(f"depth_conf.pt shape mismatch: expected {depth_shape}, got {depth_conf_shape}")

    artifacts = metadata.get("v4_artifacts")
    if artifacts is not None:
        expected = {
            "depth": "depth.pt",
            "depth_conf": "depth_conf.pt",
            "predicted_intrinsics": "predicted_intrinsics.pt",
            "transformed_gt_intrinsics": "transformed_gt_intrinsics.pt",
        }
        if not isinstance(artifacts, dict):
            errors.append("v4_artifacts must be an object when present")
        else:
            for key, artifact_name in expected.items():
                actual = artifacts.get(key)
                if actual is None:
                    errors.append(f"v4_artifacts missing key: {key}")
                    continue
                actual_path = Path(str(actual)).expanduser()
                if not actual_path.is_absolute():
                    actual_path = reconstruction_dir / actual_path
                if actual_path.resolve() != (reconstruction_dir / artifact_name).resolve():
                    errors.append(f"v4_artifacts.{key} must point to cache-local {artifact_name}")


def validate_reconstruction_cache_v4(
    reconstruction_dir: Path,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate a complete v4 reconstruction cache and return its metadata."""

    reconstruction_dir = reconstruction_dir.expanduser().resolve()
    errors: list[str] = []
    for artifact_name in REQUIRED_RECONSTRUCTION_ARTIFACTS_V4:
        artifact_path = reconstruction_dir / artifact_name
        if not artifact_path.is_file():
            errors.append(f"missing required artifact: {artifact_name}")

    metadata: dict[str, Any] = {}
    metadata_path = reconstruction_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"metadata.json is not valid JSON: {exc}")
        else:
            if not isinstance(loaded, dict):
                errors.append("metadata.json must contain a JSON object")
            else:
                metadata = loaded

    schema_version = metadata.get("cache_schema_version")
    if schema_version != CACHE_SCHEMA_VERSION_V4:
        errors.append(
            f"cache_schema_version mismatch: expected {CACHE_SCHEMA_VERSION_V4}, got {schema_version}"
        )

    actual_fingerprint = metadata.get("cache_fingerprint")
    if not isinstance(actual_fingerprint, str) or not actual_fingerprint:
        errors.append("cache_fingerprint is missing")
    elif expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        errors.append(
            "cache_fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )

    pose_enc_path_value = metadata.get("pose_enc_path")
    if pose_enc_path_value is not None:
        pose_enc_path = Path(str(pose_enc_path_value)).expanduser()
        if not pose_enc_path.is_absolute():
            pose_enc_path = reconstruction_dir / pose_enc_path
        if pose_enc_path.resolve() != (reconstruction_dir / "pose_enc.pt").resolve():
            errors.append("pose_enc_path must point to the cache-local pose_enc.pt artifact")

    if not errors:
        _validate_v4_metadata_layout(reconstruction_dir, metadata, errors)

    if errors:
        raise ReconstructionCacheValidationError(reconstruction_dir, errors)
    return metadata


def validate_reconstruction_cache(
    reconstruction_dir: Path,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate a complete v3 reconstruction cache and return its metadata."""

    reconstruction_dir = reconstruction_dir.expanduser().resolve()
    errors = []
    for artifact_name in REQUIRED_RECONSTRUCTION_ARTIFACTS:
        artifact_path = reconstruction_dir / artifact_name
        if not artifact_path.is_file():
            errors.append(f"missing required artifact: {artifact_name}")

    metadata: dict[str, Any] = {}
    metadata_path = reconstruction_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"metadata.json is not valid JSON: {exc}")
        else:
            if not isinstance(loaded, dict):
                errors.append("metadata.json must contain a JSON object")
            else:
                metadata = loaded

    schema_version = metadata.get("cache_schema_version")
    if schema_version != CACHE_SCHEMA_VERSION:
        errors.append(
            f"cache_schema_version mismatch: expected {CACHE_SCHEMA_VERSION}, got {schema_version}"
        )

    actual_fingerprint = metadata.get("cache_fingerprint")
    if not isinstance(actual_fingerprint, str) or not actual_fingerprint:
        errors.append("cache_fingerprint is missing")
    elif expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        errors.append(
            "cache_fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )

    pose_enc_path_value = metadata.get("pose_enc_path")
    if pose_enc_path_value is not None:
        pose_enc_path = Path(str(pose_enc_path_value)).expanduser()
        if not pose_enc_path.is_absolute():
            pose_enc_path = reconstruction_dir / pose_enc_path
        if pose_enc_path.resolve() != (reconstruction_dir / "pose_enc.pt").resolve():
            errors.append("pose_enc_path must point to the cache-local pose_enc.pt artifact")

    if errors:
        raise ReconstructionCacheValidationError(reconstruction_dir, errors)
    return metadata
