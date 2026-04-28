"""Reporting helpers for performance tooling."""

from __future__ import annotations

from collections import defaultdict

from trinity.perf.resource_backends import ResourceSample


def build_resource_timeline_payload(samples: list[ResourceSample]) -> dict:
    """Convert raw resource samples into a chart-friendly timeline payload."""
    timeline = [sample.to_dict() for sample in samples]
    cpu_series = [
        {"timestamp": sample.timestamp, "value": sample.cpu_percent} for sample in samples
    ]
    memory_rss_series = [
        {"timestamp": sample.timestamp, "value": sample.memory_rss_mb} for sample in samples
    ]
    memory_percent_series = [
        {"timestamp": sample.timestamp, "value": sample.memory_percent} for sample in samples
    ]

    gpu_util_series: dict[int, list[dict]] = defaultdict(list)
    gpu_memory_series: dict[int, list[dict]] = defaultdict(list)
    gpu_names: dict[int, str] = {}
    for sample in samples:
        for gpu_sample in sample.gpu_metrics:
            gpu_names[gpu_sample.gpu_id] = gpu_sample.name
            gpu_util_series[gpu_sample.gpu_id].append(
                {"timestamp": sample.timestamp, "value": gpu_sample.gpu_util_percent}
            )
            gpu_memory_series[gpu_sample.gpu_id].append(
                {"timestamp": sample.timestamp, "value": gpu_sample.gpu_memory_used_mb}
            )

    return {
        "resource_timeline": timeline,
        "chart_series": {
            "cpu_percent": cpu_series,
            "memory_rss_mb": memory_rss_series,
            "memory_percent": memory_percent_series,
            "gpu_util_percent": {
                str(gpu_id): {
                    "gpu_id": gpu_id,
                    "name": gpu_names[gpu_id],
                    "values": values,
                }
                for gpu_id, values in gpu_util_series.items()
            },
            "gpu_memory_used_mb": {
                str(gpu_id): {
                    "gpu_id": gpu_id,
                    "name": gpu_names[gpu_id],
                    "values": values,
                }
                for gpu_id, values in gpu_memory_series.items()
            },
        },
    }
