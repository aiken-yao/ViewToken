from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.extract_vggt_features import (
    extract_model_state_dict,
    load_checkpoint_state_dict,
    load_vggt_model,
)


class CheckpointLoadingTest(unittest.TestCase):
    def test_extracts_nested_model_state_dict(self) -> None:
        state_dict = {"weight": torch.ones(2)}
        self.assertIs(extract_model_state_dict({"model": state_dict}), state_dict)

    def test_loads_weights_only_and_strips_common_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"model": {"module.weight": torch.ones(2)}}, path)

            state_dict = load_checkpoint_state_dict(path)

        self.assertEqual(list(state_dict), ["weight"])
        self.assertTrue(torch.equal(state_dict["weight"], torch.ones(2)))

    def test_local_checkpoint_must_include_phase0_point_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model": {
                        "aggregator.weight": torch.ones(1),
                        "camera_head.weight": torch.ones(1),
                        "depth_head.weight": torch.ones(1),
                    }
                },
                path,
            )

            with self.assertRaisesRegex(RuntimeError, "point_head"):
                load_vggt_model("unused", path)


if __name__ == "__main__":
    unittest.main()
