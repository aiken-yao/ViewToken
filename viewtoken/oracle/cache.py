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
