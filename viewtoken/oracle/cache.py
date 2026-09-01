"""Cache fingerprints for oracle reconstruction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = "oracle-reconstruction-v3"


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
    max_points: int,
    seed: int,
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
    max_points: int,
    seed: int,
) -> dict[str, Any]:
    payload = build_reconstruction_cache_payload(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        preprocess_mode=preprocess_mode,
        layer_index=layer_index,
        min_confidence=min_confidence,
        max_points=max_points,
        seed=seed,
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": hash_cache_payload(payload),
        "payload": payload,
    }
