from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from viewtoken.oracle import build_reconstruction_cache_identity


class OracleCacheTest(unittest.TestCase):
    def test_reconstruction_cache_fingerprint_changes_with_ordered_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "model.pt"
            first = root / "00000.jpg"
            second = root / "00010.jpg"
            checkpoint.write_bytes(b"weights")
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            forward = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[first, second],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=50000,
                seed=0,
            )
            reversed_order = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[second, first],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=50000,
                seed=0,
            )

        self.assertNotEqual(forward["fingerprint"], reversed_order["fingerprint"])

    def test_reconstruction_cache_fingerprint_changes_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "model.pt"
            image = root / "00000.jpg"
            checkpoint.write_bytes(b"weights")
            image.write_bytes(b"image")

            base = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=50000,
                seed=0,
            )
            changed = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=1.0,
                max_points=50000,
                seed=0,
            )

        self.assertNotEqual(base["fingerprint"], changed["fingerprint"])
        self.assertEqual(base["schema_version"], "oracle-reconstruction-v3")


if __name__ == "__main__":
    unittest.main()
