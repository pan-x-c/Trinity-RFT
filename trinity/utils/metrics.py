"""Unified metrics aggregation utilities for Trinity-RFT.

Metric keys may carry an aggregation-type suffix in the form ``name:agg``.
Supported suffixes: ``:mean``, ``:sum``, ``:max``, ``:min``, ``:last``.
Keys without a suffix default to mean aggregation.
"""

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


class AggType(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


def parse_metric_key(key: str) -> Tuple[str, AggType]:
    """Parse a metric key into (name, aggregation_type).

    Examples:
        "reward"               -> ("reward", AggType.MEAN)
        "experience_count:sum" -> ("experience_count", AggType.SUM)
        "model_version:last"   -> ("model_version", AggType.LAST)
        "some:unknown_suffix"  -> ("some:unknown_suffix", AggType.MEAN)
    """
    if ":" in key:
        name, agg_str = key.rsplit(":", 1)
        try:
            return name, AggType(agg_str)
        except ValueError:
            return key, AggType.MEAN
    return key, AggType.MEAN


def aggregate_metrics(
    metric_dicts: List[Dict[str, float]],
    prefix: str = "",
    default_output_stats: List[str] | None = None,
) -> Dict[str, float]:
    """Aggregate a list of metric dictionaries respecting per-key aggregation types.

    For keys with AggType.MEAN, outputs ``{prefix}/{name}/mean``, ``/max``, ``/min``
    (controlled by *default_output_stats*).
    For AggType.SUM, outputs ``{prefix}/{name}/sum``.
    For AggType.MAX, outputs ``{prefix}/{name}/max``.
    For AggType.MIN, outputs ``{prefix}/{name}/min``.
    For AggType.LAST, outputs ``{prefix}/{name}`` (the last observed value).

    Args:
        metric_dicts: List of flat metric dictionaries (values must be numeric).
        prefix: Optional prefix prepended as ``{prefix}/{name}/...``.
        default_output_stats: Stats to output for MEAN metrics. Defaults to ["mean", "max", "min"].

    Returns:
        Flat dictionary of aggregated metrics ready for monitor logging.
    """
    if not metric_dicts:
        return {}

    if default_output_stats is None:
        default_output_stats = ["mean", "max", "min"]

    # Collect values grouped by (parsed_name, agg_type)
    grouped: Dict[Tuple[str, AggType], List[float]] = defaultdict(list)
    for d in metric_dicts:
        for key, value in d.items():
            if not isinstance(value, (int, float)):
                continue
            name, agg = parse_metric_key(key)
            grouped[(name, agg)].append(float(value))

    result: Dict[str, float] = {}
    prefix_str = f"{prefix}/" if prefix else ""

    for (name, agg), values in grouped.items():
        if agg == AggType.MEAN:
            arr = np.array(values)
            stat_fns = {
                "mean": np.mean,
                "max": np.max,
                "min": np.min,
                "std": np.std,
            }
            for stat in default_output_stats:
                fn = stat_fns.get(stat)
                if fn is not None:
                    result[f"{prefix_str}{name}/{stat}"] = fn(arr).item()
        elif agg == AggType.SUM:
            result[f"{prefix_str}{name}/sum"] = sum(values)
        elif agg == AggType.MAX:
            result[f"{prefix_str}{name}/max"] = max(values)
        elif agg == AggType.MIN:
            result[f"{prefix_str}{name}/min"] = min(values)
        elif agg == AggType.LAST:
            result[f"{prefix_str}{name}"] = values[-1]

    return result


def aggregate_eval_metrics(
    metric_dicts: List[Dict[str, float]],
    prefix: str = "",
    output_stats: List[str] | None = None,
    detailed_stats: bool = False,
) -> Dict[str, float]:
    """Aggregate eval metrics with optional detailed statistics.

    For MEAN metrics:
      - If detailed_stats=True: output mean/max/min/std per the output_stats list.
      - If detailed_stats=False: output only the mean value as ``{prefix}/{name}``.
    For non-MEAN metrics: same behavior as aggregate_metrics.
    """
    if not metric_dicts:
        return {}

    if output_stats is None:
        output_stats = ["mean", "max", "min", "std"]

    grouped: Dict[Tuple[str, AggType], List[float]] = defaultdict(list)
    for d in metric_dicts:
        for key, value in d.items():
            if not isinstance(value, (int, float)):
                continue
            name, agg = parse_metric_key(key)
            grouped[(name, agg)].append(float(value))

    result: Dict[str, float] = {}
    prefix_str = f"{prefix}/" if prefix else ""

    for (name, agg), values in grouped.items():
        if agg == AggType.MEAN:
            arr = np.array(values)
            if detailed_stats:
                stat_fns = {
                    "mean": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "std": np.std,
                }
                for stat in output_stats:
                    fn = stat_fns.get(stat)
                    if fn is not None:
                        result[f"{prefix_str}{name}/{stat}"] = fn(arr).item()
            else:
                result[f"{prefix_str}{name}"] = np.mean(arr).item()
        elif agg == AggType.SUM:
            result[f"{prefix_str}{name}/sum"] = sum(values)
        elif agg == AggType.MAX:
            result[f"{prefix_str}{name}/max"] = max(values)
        elif agg == AggType.MIN:
            result[f"{prefix_str}{name}/min"] = min(values)
        elif agg == AggType.LAST:
            result[f"{prefix_str}{name}"] = values[-1]

    return result


def aggregate_run_level_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate experience-level metrics into a single run-level metric dict.

    Unlike batch-level aggregation, this preserves the original key format (with ``:agg``
    suffix if present) so that downstream task/batch aggregation can still see the
    aggregation type annotation.

    Aggregation rules:
      - MEAN keys: averaged across experiences
      - SUM keys: summed across experiences
      - MAX keys: max across experiences
      - MIN keys: min across experiences
      - LAST keys: last value
    """
    if not metric_dicts:
        return {}

    grouped: Dict[str, Tuple[AggType, List[float]]] = {}
    for d in metric_dicts:
        for key, value in d.items():
            if not isinstance(value, (int, float)):
                continue
            name, agg = parse_metric_key(key)
            canonical_key = f"{name}:{agg.value}" if agg != AggType.MEAN else name
            if canonical_key not in grouped:
                grouped[canonical_key] = (agg, [])
            grouped[canonical_key][1].append(float(value))

    result: Dict[str, float] = {}
    for key, (agg, values) in grouped.items():
        if agg == AggType.MEAN:
            result[key] = sum(values) / len(values)
        elif agg == AggType.SUM:
            result[key] = sum(values)
        elif agg == AggType.MAX:
            result[key] = max(values)
        elif agg == AggType.MIN:
            result[key] = min(values)
        elif agg == AggType.LAST:
            result[key] = values[-1]

    return result
