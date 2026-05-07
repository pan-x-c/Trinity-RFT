from __future__ import annotations

from typing import Any, Optional

EXPERIENCE_COUNT_METRIC_KEY = "experience_pipeline/experience_count"
PROMPT_LENGTH_MEAN_METRIC_KEY = "rollout/prompt_length/mean"
RESPONSE_LENGTH_MEAN_METRIC_KEY = "rollout/response_length/mean"


def compute_global_token_throughput_metrics(
    execution_time_sec: Optional[float], step_metrics: list[dict[str, Any]]
) -> dict[str, float | None]:
    if execution_time_sec is None or execution_time_sec <= 0:
        return {
            "prompt_tokens_per_sec": None,
            "response_tokens_per_sec": None,
        }

    prompt_token_total = 0.0
    response_token_total = 0.0
    for step_metric in step_metrics:
        experience_count = step_metric.get(EXPERIENCE_COUNT_METRIC_KEY)
        prompt_length_mean = step_metric.get(PROMPT_LENGTH_MEAN_METRIC_KEY)
        response_length_mean = step_metric.get(RESPONSE_LENGTH_MEAN_METRIC_KEY)
        if experience_count is None:
            continue
        if prompt_length_mean is not None:
            prompt_token_total += float(experience_count) * float(prompt_length_mean)
        if response_length_mean is not None:
            response_token_total += float(experience_count) * float(response_length_mean)

    return {
        "prompt_tokens_per_sec": prompt_token_total / float(execution_time_sec),
        "response_tokens_per_sec": response_token_total / float(execution_time_sec),
    }
