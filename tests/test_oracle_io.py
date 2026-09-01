from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from viewtoken.oracle.io import infer_bin_stride, load_point_cloud, load_pose_matrix


class OracleIOTest(unittest.TestCase):
    def test_infers_scannet_xyzrgb_bin_stride(self) -> None:
        values = np.array([1, 2, 3, 80, 90, 100, 4, 5, 6, 10, 20, 30], dtype=np.float32)
        self.assertEqual(infer_bin_stride(values), 6)

    def test_loads_binary_xyzrgb_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "points.bin"
            np.array([1, 2, 3, 80, 90, 100, 4, 5, 6, 10, 20, 30], dtype=np.float32).tofile(path)

            points = load_point_cloud(path)

        self.assertEqual(tuple(points.shape), (2, 3))
        self.assertEqual(points[1].tolist(), [4.0, 5.0, 6.0])

    def test_loads_pose_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pose.txt"
            np.savetxt(path, np.eye(4, dtype=np.float32))

            pose = load_pose_matrix(path)

        self.assertEqual(len(pose), 4)
        self.assertEqual(pose[3], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
