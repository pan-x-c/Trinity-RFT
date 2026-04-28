"""Helpers for TensorBoard metric parsing and aggregation."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEP_TIME_METRIC_CANDIDATES = (
    "time/wait_explore_step",
    "time/explorer_sync_interval",
)
FINISHED_TASK_METRIC_NAME = "rollout/finished_task_count"


class TensorBoardScalarReader:
    """Read scalar metrics from TensorBoard event files."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = self._load_metrics(log_dir)

    def _load_metrics(self, log_dir: str) -> dict[str, dict[int, float]]:
        metric_map: dict[str, dict[int, float]] = defaultdict(dict)
        for event_file in self._find_event_files(log_dir):
            accumulator = EventAccumulator(event_file)
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for scalar in accumulator.Scalars(tag):
                    prior_value = metric_map[tag].get(scalar.step)
                    if prior_value is None or scalar.value > prior_value:
                        metric_map[tag][scalar.step] = scalar.value
        return dict(metric_map)

    def _find_event_files(self, log_dir: str) -> list[str]:
        event_files: list[str] = []
        for root, _, files in os.walk(log_dir):
            for file_name in files:
                if file_name.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, file_name))
        return sorted(event_files)


def get_step_time(metric_map: dict[str, dict[int, float]], step: int) -> Optional[float]:
    """Select the best available step duration metric for one step."""
    for metric_name in STEP_TIME_METRIC_CANDIDATES:
        if step in metric_map.get(metric_name, {}):
            return float(metric_map[metric_name][step])
    return None


def extract_raw_metrics_for_step(
    metric_map: dict[str, dict[int, float]], step: int
) -> dict[str, float]:
    """Extract all scalar metrics that were logged for one step."""
    return {
        metric_name: float(step_values[step])
        for metric_name, step_values in metric_map.items()
        if step in step_values
    }


def collect_step_metrics(metric_map: dict[str, dict[int, float]]) -> list[dict[str, Any]]:
    """Build per-step metrics from TensorBoard scalars."""
    step_numbers = sorted(metric_map.get(FINISHED_TASK_METRIC_NAME, {}).keys())
    step_metrics: list[dict[str, Any]] = []
    for step in step_numbers:
        finished_task_count = float(metric_map[FINISHED_TASK_METRIC_NAME][step])
        step_time_sec = get_step_time(metric_map, step)
        step_metrics.append(
            {
                "step": step,
                "finished_task_count": finished_task_count,
                "step_time_sec": step_time_sec,
                "throughput_task_per_min": (
                    finished_task_count / step_time_sec * 60.0
                    if step_time_sec is not None and step_time_sec > 0 and finished_task_count > 0
                    else None
                ),
                "avg_task_time_sec": (
                    step_time_sec / finished_task_count
                    if step_time_sec is not None and step_time_sec > 0 and finished_task_count > 0
                    else None
                ),
                "raw_metrics": extract_raw_metrics_for_step(metric_map, step),
            }
        )
    return step_metrics


def build_global_metrics(step_metrics: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Aggregate global metrics from per-step records."""
    total_finished_task_count = float(
        sum(step_metric["finished_task_count"] for step_metric in step_metrics)
    )
    total_step_time_sec = sum(
        step_metric["step_time_sec"]
        for step_metric in step_metrics
        if step_metric["step_time_sec"] is not None
    )
    if total_finished_task_count > 0 and total_step_time_sec > 0:
        overall_throughput = total_finished_task_count / total_step_time_sec * 60.0
        overall_avg_task_time = total_step_time_sec / total_finished_task_count
    else:
        overall_throughput = None
        overall_avg_task_time = None
    return {
        "total_finished_task_count": total_finished_task_count,
        "overall_throughput_task_per_min": overall_throughput,
        "overall_avg_task_time_sec": overall_avg_task_time,
        "total_step_time_sec": total_step_time_sec if total_step_time_sec > 0 else None,
    }
