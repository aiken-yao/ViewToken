#!/usr/bin/env python3
"""Validate and audit the six fixed Stage C5 v4 caches."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from viewtoken.oracle import load_v4_cache_data, cache_artifact_shape_summary

OBSERVED = ["00000", "00010", "00020"]
CANDIDATES = ["00018", "00369", "00384", "00065", "00437"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("outputs/oracle_calibration/scannet_scene0000_00_stage_c5_v4_smoke/reconstructions"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    expected = {"baseline": OBSERVED, **{c: OBSERVED + [c] for c in CANDIDATES}}
    rows=[]; missing=[]; invalid=[]
    for label, ids in expected.items():
        suffix = "" if label == "baseline" else f"__plus__{label}"
        matches = sorted(args.cache_root.glob("*" if label == "baseline" else f"*{suffix}"))
        if label == "baseline":
            matches = [path for path in matches if "__plus__" not in path.name]
        if not matches:
            missing.append({"label": label, "expected_view_ids": ids, "expected_directory_suffix": suffix})
            continue
        if len(matches) != 1:
            invalid.append({"label": label, "error": "ambiguous_cache_directories", "matches": [str(x) for x in matches]})
            continue
        try:
            cache = load_v4_cache_data(matches[0], expected_view_ids=ids)
            rows.append({"label": label, "status": "valid", "artifacts": cache_artifact_shape_summary(cache)})
        except Exception as exc:
            invalid.append({"label": label, "error": str(exc), "path": str(matches[0])})
    result = {"stage": "C5", "expected_cache_count": 6, "validated_cache_count": len(rows), "missing": missing, "invalid": invalid, "status": "blocked_missing_v4_cache" if missing else ("failed_cache_validation" if invalid else "validated") , "caches": rows}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(2 if missing or invalid else 0)
if __name__ == "__main__": main()
