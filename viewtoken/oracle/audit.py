"""Audit summaries for oracle reconstruction-gain candidates."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from .dataset import OracleGainRecord

SANITY_EXPECTATIONS = {
    "identical_cloud": "The same cached cloud evaluated twice should have zero gain up to floating-point noise.",
    "duplicate_input_sensitivity": "Repeating an observed RGB through VGGT measures model sensitivity and is not a legal NBV candidate.",
    "high_overlap_neighbor": "Gain is expected to be small for views with high overlap to observed inputs.",
    "new_area": "At least one coverage or completeness-oriented metric should improve if new surfaces are observed.",
}


def flatten_metric_gains(record: OracleGainRecord) -> dict[str, float]:
    gains = record.gains
    flat = {
        "chamfer": float(gains["chamfer"]),
        "accuracy": float(gains["accuracy"]),
        "completeness": float(gains["completeness"]),
        "coverage": float(gains["coverage"]),
    }
    for threshold, value in gains["fscore"].items():
        flat[f"fscore@{threshold}"] = float(value)
    return flat


def _metric_sort_key(name: str) -> tuple[int, float | str]:
    order = {
        "chamfer": 0,
        "accuracy": 1,
        "completeness": 2,
        "coverage": 3,
    }
    if name.startswith("fscore@"):
        return 4, float(name.split("@", 1)[1])
    return order.get(name, 99), name


def _ordered_metric_names(records: list[OracleGainRecord]) -> list[str]:
    names: set[str] = set()
    for record in records:
        names.update(flatten_metric_gains(record))
    return sorted(names, key=_metric_sort_key)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _value in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def spearman_rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have the same length")
    if len(left) < 2:
        return None

    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_denominator = math.sqrt(
        sum((left_value - left_mean) ** 2 for left_value in left_ranks)
    )
    right_denominator = math.sqrt(
        sum((right_value - right_mean) ** 2 for right_value in right_ranks)
    )
    denominator = left_denominator * right_denominator
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_metric_gains(records: list[OracleGainRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric_name in _ordered_metric_names(records):
        pairs = [
            (record.candidate_view_id, flatten_metric_gains(record)[metric_name])
            for record in records
            if metric_name in flatten_metric_gains(record)
        ]
        if not pairs:
            continue
        values = [value for _candidate_id, value in pairs]
        best_candidate, best_gain = max(pairs, key=lambda item: item[1])
        summary[metric_name] = {
            "candidate_count": len(values),
            "min": min(values),
            "median": median(values),
            "mean": mean(values),
            "max": max(values),
            "positive_gain_ratio": sum(value > 0 for value in values) / len(values),
            "oracle_best_candidate": best_candidate,
            "oracle_best_gain": best_gain,
            "random_candidate_mean_gain": mean(values),
        }
    return summary


def spearman_rank_correlations(records: list[OracleGainRecord]) -> dict[str, float | None]:
    metric_names = _ordered_metric_names(records)
    gains_by_record = [flatten_metric_gains(record) for record in records]
    correlations: dict[str, float | None] = {}
    for left_index, left_name in enumerate(metric_names):
        for right_name in metric_names[left_index + 1 :]:
            paired = [
                (gains[left_name], gains[right_name])
                for gains in gains_by_record
                if left_name in gains and right_name in gains
            ]
            if not paired:
                correlations[f"{left_name}|{right_name}"] = None
                continue
            left_values, right_values = zip(*paired, strict=True)
            correlations[f"{left_name}|{right_name}"] = spearman_rank_correlation(
                list(left_values), list(right_values)
            )
    return correlations


def _record_is_held_out(record: OracleGainRecord, observed_view_ids: set[str]) -> bool:
    return record.candidate_view_id not in observed_view_ids


def summarize_sanity_checks(records: list[OracleGainRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for tag, expectation in SANITY_EXPECTATIONS.items():
        tagged_records = [
            record
            for record in records
            if tag in record.metadata.get("candidate_sanity_tags", [])
        ]
        summary[tag] = {
            "candidate_count": len(tagged_records),
            "expected_behavior": expectation,
            "entries": [
                {
                    "candidate_view_id": record.candidate_view_id,
                    "pose_min_distance_to_observed_meters": record.metadata.get(
                        "pose_min_distance_to_observed_meters"
                    ),
                    "gains": flatten_metric_gains(record),
                }
                for record in tagged_records
            ],
        }
    return summary


def summarize_oracle_audit(
    records: list[OracleGainRecord], observed_view_ids: list[str]
) -> dict[str, Any]:
    observed = set(observed_view_ids)
    held_out_records = [record for record in records if _record_is_held_out(record, observed)]
    return {
        "candidate_count": len(records),
        "held_out_candidate_count": len(held_out_records),
        "metrics": summarize_metric_gains(records),
        "held_out_metrics": summarize_metric_gains(held_out_records),
        "spearman_rank_correlation": spearman_rank_correlations(records),
        "held_out_spearman_rank_correlation": spearman_rank_correlations(held_out_records),
        "sanity_checks": summarize_sanity_checks(records),
    }
