from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from viewtoken.oracle import (
    CACHE_SCHEMA_VERSION_V4,
    ReconstructionCacheValidationError,
    build_reconstruction_cache_identity,
    build_v4_reconstruction_cache_identity,
    validate_reconstruction_cache,
    validate_reconstruction_cache_v4,
)


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


    def test_reconstruction_cache_fingerprint_changes_with_sample_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "model.pt"
            image = root / "00000.jpg"
            checkpoint.write_bytes(b"weights")
            image.write_bytes(b"image")

            random_sample = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=50000,
                seed=0,
                sample_method="random",
            )
            no_sample = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=None,
                seed=0,
                sample_method="none",
            )

        self.assertNotEqual(random_sample["fingerprint"], no_sample["fingerprint"])
        self.assertEqual(
            no_sample["payload"]["reconstruction_sample_method"],
            "none",
        )

    def test_validate_reconstruction_cache_requires_v3_artifacts_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "model.pt"
            image = root / "00000.jpg"
            cache_dir = root / "cache"
            cache_dir.mkdir()
            checkpoint.write_bytes(b"weights")
            image.write_bytes(b"image")
            identity = build_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=50000,
                seed=0,
            )
            for name in ("points.pt", "confidence.pt", "pose_enc.pt"):
                (cache_dir / name).write_bytes(b"artifact")
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "cache_schema_version": identity["schema_version"],
                        "cache_fingerprint": identity["fingerprint"],
                        "pose_enc_path": str(cache_dir / "pose_enc.pt"),
                    }
                )
            )

            metadata = validate_reconstruction_cache(
                cache_dir, expected_fingerprint=identity["fingerprint"]
            )

        self.assertEqual(metadata["cache_fingerprint"], identity["fingerprint"])

    def test_validate_reconstruction_cache_rejects_missing_pose_enc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            (cache_dir / "points.pt").write_bytes(b"points")
            (cache_dir / "confidence.pt").write_bytes(b"confidence")
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "cache_schema_version": "oracle-reconstruction-v3",
                        "cache_fingerprint": "abc",
                    }
                )
            )

            with self.assertRaises(ReconstructionCacheValidationError) as context:
                validate_reconstruction_cache(cache_dir, expected_fingerprint="abc")

        self.assertIn("missing required artifact: pose_enc.pt", context.exception.errors)

    def test_validate_reconstruction_cache_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            for name in ("points.pt", "confidence.pt", "pose_enc.pt"):
                (cache_dir / name).write_bytes(b"artifact")
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "cache_schema_version": "oracle-reconstruction-v3",
                        "cache_fingerprint": "actual",
                        "pose_enc_path": str(cache_dir / "pose_enc.pt"),
                    }
                )
            )

            with self.assertRaises(ReconstructionCacheValidationError) as context:
                validate_reconstruction_cache(cache_dir, expected_fingerprint="expected")

        self.assertTrue(
            any("cache_fingerprint mismatch" in error for error in context.exception.errors)
        )

    def write_complete_v4_cache(self, cache_dir: Path, fingerprint: str = "fingerprint") -> None:
        cache_dir.mkdir()
        torch.save(torch.zeros((12, 3), dtype=torch.float32), cache_dir / "points.pt")
        torch.save(torch.ones((12,), dtype=torch.float32), cache_dir / "confidence.pt")
        torch.save(torch.zeros((1, 2, 9), dtype=torch.float32), cache_dir / "pose_enc.pt")
        torch.save(torch.ones((2, 2, 3), dtype=torch.float32), cache_dir / "depth.pt")
        torch.save(torch.ones((2, 2, 3), dtype=torch.float32), cache_dir / "depth_conf.pt")
        torch.save(torch.eye(3, dtype=torch.float32).repeat(2, 1, 1), cache_dir / "predicted_intrinsics.pt")
        torch.save(torch.eye(3, dtype=torch.float32).repeat(2, 1, 1), cache_dir / "transformed_gt_intrinsics.pt")
        (cache_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_schema_version": CACHE_SCHEMA_VERSION_V4,
                    "cache_fingerprint": fingerprint,
                    "pose_enc_path": str(cache_dir / "pose_enc.pt"),
                    "image_paths": ["00000.jpg", "00010.jpg"],
                    "input_shape": [2, 3, 2, 3],
                    "preprocessing_transforms": [
                        {"mode": "crop", "output_width": 3, "output_height": 2},
                        {"mode": "crop", "output_width": 3, "output_height": 2},
                    ],
                    "per_view_shape_offsets": [
                        {"view_id": "00000", "height": 2, "width": 3, "point_offset": 0, "point_count": 6},
                        {"view_id": "00010", "height": 2, "width": 3, "point_offset": 6, "point_count": 6},
                    ],
                    "v4_artifacts": {
                        "depth": "depth.pt",
                        "depth_conf": "depth_conf.pt",
                        "predicted_intrinsics": "predicted_intrinsics.pt",
                        "transformed_gt_intrinsics": "transformed_gt_intrinsics.pt",
                    },
                }
            )
        )

    def test_v4_reconstruction_cache_fingerprint_changes_with_preprocessing_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "model.pt"
            image = root / "00000.jpg"
            checkpoint.write_bytes(b"weights")
            image.write_bytes(b"image")

            base = build_v4_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=None,
                seed=0,
                preprocessing_transforms=[{"scale_x": 1.0, "scale_y": 1.0}],
                per_view_shape_offsets=[{"height": 2, "width": 3, "point_offset": 0, "point_count": 6}],
            )
            changed = build_v4_reconstruction_cache_identity(
                checkpoint_path=checkpoint,
                image_paths=[image],
                preprocess_mode="crop",
                layer_index=23,
                min_confidence=0.0,
                max_points=None,
                seed=0,
                preprocessing_transforms=[{"scale_x": 0.5, "scale_y": 1.0}],
                per_view_shape_offsets=[{"height": 2, "width": 3, "point_offset": 0, "point_count": 6}],
            )

        self.assertEqual(base["schema_version"], CACHE_SCHEMA_VERSION_V4)
        self.assertNotEqual(base["fingerprint"], changed["fingerprint"])

    def test_validate_reconstruction_cache_v4_accepts_complete_depth_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            self.write_complete_v4_cache(cache_dir, fingerprint="abc")

            metadata = validate_reconstruction_cache_v4(cache_dir, expected_fingerprint="abc")

        self.assertEqual(metadata["cache_schema_version"], CACHE_SCHEMA_VERSION_V4)

    def test_validate_reconstruction_cache_v4_rejects_missing_depth_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            self.write_complete_v4_cache(cache_dir, fingerprint="abc")
            (cache_dir / "depth.pt").unlink()

            with self.assertRaises(ReconstructionCacheValidationError) as context:
                validate_reconstruction_cache_v4(cache_dir, expected_fingerprint="abc")

        self.assertIn("missing required artifact: depth.pt", context.exception.errors)

    def test_validate_reconstruction_cache_v4_rejects_noncontiguous_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            self.write_complete_v4_cache(cache_dir, fingerprint="abc")
            metadata = json.loads((cache_dir / "metadata.json").read_text())
            metadata["per_view_shape_offsets"][1]["point_offset"] = 7
            (cache_dir / "metadata.json").write_text(json.dumps(metadata))

            with self.assertRaises(ReconstructionCacheValidationError) as context:
                validate_reconstruction_cache_v4(cache_dir, expected_fingerprint="abc")

        self.assertTrue(any("point_offset mismatch" in error for error in context.exception.errors))


if __name__ == "__main__":
    unittest.main()
