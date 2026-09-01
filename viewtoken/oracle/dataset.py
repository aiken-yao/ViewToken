"""Serializable records for oracle reconstruction-gain datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .metrics import PointCloudMetrics, compute_metric_gains


@dataclass(frozen=True)
class OracleGainRecord:
    scene_id: str
    split: str
    memory_id: str
    observed_view_ids: list[str]
    candidate_view_id: str
    candidate_pose: list[list[float]]
    baseline_metrics: PointCloudMetrics
    candidate_metrics: PointCloudMetrics
    reconstruction_paths: dict[str, str]
    metadata: dict[str, Any]

    @property
    def gains(self) -> dict[str, object]:
        return compute_metric_gains(self.baseline_metrics, self.candidate_metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "split": self.split,
            "memory_id": self.memory_id,
            "observed_view_ids": self.observed_view_ids,
            "candidate_view_id": self.candidate_view_id,
            "candidate_pose": self.candidate_pose,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "gains": self.gains,
            "reconstruction_paths": self.reconstruction_paths,
            "metadata": self.metadata,
        }


def build_memory_id(scene_id: str, observed_view_ids: list[str]) -> str:
    payload = json.dumps(
        {"scene_id": scene_id, "observed_view_ids": observed_view_ids},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def scene_split(
    scene_id: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    salt: str = "viewtoken-v1",
) -> str:
    if not 0 <= train_fraction <= 1 or not 0 <= val_fraction <= 1:
        raise ValueError("split fractions must be in [0, 1]")
    if train_fraction + val_fraction > 1:
        raise ValueError("train_fraction + val_fraction must be <= 1")

    digest = hashlib.sha1(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "val"
    return "test"


def _write_records(handle: TextIO, records: list[OracleGainRecord]) -> None:
    for record in records:
        handle.write(json.dumps(record.to_dict(), sort_keys=True))
        handle.write("\n")


def write_jsonl(path: Path, records: list[OracleGainRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        _write_records(handle, records)


def append_jsonl(path: Path, records: list[OracleGainRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        _write_records(handle, records)
