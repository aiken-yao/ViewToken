"""Input helpers for oracle-gain generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def load_pose_matrix(path: Path) -> list[list[float]]:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix in {path}, got {pose.shape}")
    return pose.tolist()


def _tensor_from_checkpoint(value: Any, path: Path) -> Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("points", "xyz", "point_cloud"):
            tensor = value.get(key)
            if torch.is_tensor(tensor):
                return tensor
    raise TypeError(f"Could not find a point tensor in {path}")


def infer_bin_stride(values: np.ndarray) -> int:
    if values.size % 6 == 0:
        maybe_color = values.reshape(-1, 6)[:, 3:6]
        finite = np.isfinite(maybe_color).all()
        if finite and maybe_color.size and maybe_color.min() >= 0 and maybe_color.max() <= 255:
            if np.median(np.abs(maybe_color)) > 1:
                return 6
    if values.size % 3 == 0:
        return 3
    raise ValueError("Binary point cloud float count is not divisible by 3 or 6")


def load_point_cloud(path: Path, point_stride: int | None = None) -> Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt" or suffix == ".pth":
        return _tensor_from_checkpoint(torch.load(path, map_location="cpu", weights_only=True), path).float()
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("points", "xyz", "point_cloud"):
            if key in payload:
                return torch.from_numpy(payload[key]).float()
        raise KeyError(f"No points/xyz/point_cloud array found in {path}")
    if suffix == ".bin":
        values = np.fromfile(path, dtype=np.float32)
        stride = point_stride or infer_bin_stride(values)
        if stride < 3 or values.size % stride != 0:
            raise ValueError(f"Cannot reshape {path} with stride {stride}")
        return torch.from_numpy(values.reshape(-1, stride)[:, :3]).float()
    raise ValueError(f"Unsupported point-cloud extension: {path.suffix}")


def view_id_from_path(path: Path) -> str:
    return path.stem
