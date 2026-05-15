"""Tests for trinity.utils.metrics aggregation utilities."""

import pytest

from trinity.utils.metrics import (
    AggType,
    aggregate_eval_metrics,
    aggregate_metrics,
    aggregate_run_level_metrics,
    parse_metric_key,
)


class TestParseMetricKey:
    def test_plain_key(self):
        name, agg = parse_metric_key("reward")
        assert name == "reward"
        assert agg == AggType.MEAN

    def test_sum_suffix(self):
        name, agg = parse_metric_key("experience_count:sum")
        assert name == "experience_count"
        assert agg == AggType.SUM

    def test_max_suffix(self):
        name, agg = parse_metric_key("latency:max")
        assert name == "latency"
        assert agg == AggType.MAX

    def test_min_suffix(self):
        name, agg = parse_metric_key("latency:min")
        assert name == "latency"
        assert agg == AggType.MIN

    def test_last_suffix(self):
        name, agg = parse_metric_key("model_version:last")
        assert name == "model_version"
        assert agg == AggType.LAST

    def test_mean_suffix_explicit(self):
        name, agg = parse_metric_key("reward:mean")
        assert name == "reward"
        assert agg == AggType.MEAN

    def test_unknown_suffix_treated_as_full_key(self):
        name, agg = parse_metric_key("some:unknown_suffix")
        assert name == "some:unknown_suffix"
        assert agg == AggType.MEAN

    def test_slashed_key_with_suffix(self):
        name, agg = parse_metric_key("time/task_execution:sum")
        assert name == "time/task_execution"
        assert agg == AggType.SUM

    def test_key_with_multiple_colons(self):
        name, agg = parse_metric_key("ns:metric_name:sum")
        assert name == "ns:metric_name"
        assert agg == AggType.SUM


class TestAggregateMetrics:
    def test_empty_input(self):
        assert aggregate_metrics([]) == {}

    def test_single_dict_mean_default(self):
        result = aggregate_metrics([{"reward": 1.0}], prefix="rollout")
        assert result == {
            "rollout/reward/mean": 1.0,
            "rollout/reward/max": 1.0,
            "rollout/reward/min": 1.0,
        }

    def test_multiple_dicts_mean(self):
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result["rollout/reward/mean"] == 2.0
        assert result["rollout/reward/max"] == 3.0
        assert result["rollout/reward/min"] == 1.0

    def test_sum_aggregation(self):
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/tokens/sum": 300.0}

    def test_max_aggregation(self):
        dicts = [{"latency:max": 5.0}, {"latency:max": 10.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/latency/max": 10.0}

    def test_min_aggregation(self):
        dicts = [{"latency:min": 5.0}, {"latency:min": 2.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/latency/min": 2.0}

    def test_last_aggregation(self):
        dicts = [{"model_version:last": 1.0}, {"model_version:last": 3.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/model_version": 3.0}

    def test_mixed_agg_types(self):
        dicts = [
            {"reward": 1.0, "tokens:sum": 100.0, "version:last": 1.0},
            {"reward": 3.0, "tokens:sum": 200.0, "version:last": 2.0},
        ]
        result = aggregate_metrics(dicts, prefix="r")
        assert result["r/reward/mean"] == 2.0
        assert result["r/tokens/sum"] == 300.0
        assert result["r/version"] == 2.0

    def test_no_prefix(self):
        dicts = [{"reward": 2.0}]
        result = aggregate_metrics(dicts, prefix="")
        assert result == {"reward/mean": 2.0, "reward/max": 2.0, "reward/min": 2.0}

    def test_custom_output_stats(self):
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_metrics(dicts, prefix="p", default_output_stats=["mean", "std"])
        assert "p/reward/mean" in result
        assert "p/reward/std" in result
        assert "p/reward/max" not in result

    def test_non_numeric_values_ignored(self):
        dicts = [{"reward": 1.0, "name": "hello"}]
        result = aggregate_metrics(dicts, prefix="p")
        assert "p/name/mean" not in result
        assert "p/reward/mean" in result


class TestAggregateEvalMetrics:
    def test_empty_input(self):
        assert aggregate_eval_metrics([]) == {}

    def test_simple_mean_no_detailed(self):
        dicts = [{"accuracy": 0.8}, {"accuracy": 0.9}]
        result = aggregate_eval_metrics(dicts, prefix="eval/gsm8k", detailed_stats=False)
        assert result == {"eval/gsm8k/accuracy": pytest.approx(0.85)}

    def test_detailed_stats(self):
        dicts = [{"accuracy": 0.8}, {"accuracy": 0.9}]
        result = aggregate_eval_metrics(dicts, prefix="eval/gsm8k", detailed_stats=True)
        assert "eval/gsm8k/accuracy/mean" in result
        assert "eval/gsm8k/accuracy/std" in result
        assert result["eval/gsm8k/accuracy/mean"] == pytest.approx(0.85)

    def test_sum_in_eval(self):
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_eval_metrics(dicts, prefix="eval/test", detailed_stats=False)
        assert result == {"eval/test/tokens/sum": 300.0}

    def test_last_in_eval(self):
        dicts = [{"version:last": 1.0}, {"version:last": 5.0}]
        result = aggregate_eval_metrics(dicts, prefix="eval/test", detailed_stats=False)
        assert result == {"eval/test/version": 5.0}


class TestAggregateRunLevelMetrics:
    def test_empty_input(self):
        assert aggregate_run_level_metrics([]) == {}

    def test_mean_keys_averaged(self):
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"reward": 2.0}

    def test_sum_keys_summed(self):
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"tokens:sum": 300.0}

    def test_max_keys(self):
        dicts = [{"latency:max": 5.0}, {"latency:max": 10.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"latency:max": 10.0}

    def test_last_keys(self):
        dicts = [{"version:last": 1.0}, {"version:last": 2.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"version:last": 2.0}

    def test_preserves_suffix_for_downstream(self):
        dicts = [{"reward": 1.0, "total_steps:sum": 5.0}]
        result = aggregate_run_level_metrics(dicts)
        assert "reward" in result
        assert "total_steps:sum" in result

    def test_mixed(self):
        dicts = [
            {"reward": 1.0, "steps:sum": 3.0, "peak:max": 10.0},
            {"reward": 3.0, "steps:sum": 4.0, "peak:max": 8.0},
        ]
        result = aggregate_run_level_metrics(dicts)
        assert result["reward"] == 2.0
        assert result["steps:sum"] == 7.0
        assert result["peak:max"] == 10.0


class TestBackwardCompatibility:
    """Ensure existing plain-float metric dicts work unchanged."""

    def test_workflow_metrics_unchanged(self):
        exp_metrics = [
            {"reward": 0.5, "actual_env_steps": 3.0, "time/run_execution": 1.2},
            {"reward": 0.8, "actual_env_steps": 5.0, "time/run_execution": 1.5},
        ]
        # Run level aggregation (experience → run)
        run_result = aggregate_run_level_metrics(exp_metrics)
        assert run_result["reward"] == pytest.approx(0.65)
        assert run_result["actual_env_steps"] == pytest.approx(4.0)
        assert run_result["time/run_execution"] == pytest.approx(1.35)

    def test_batch_level_aggregation_unchanged(self):
        task_metrics = [
            {"reward": 0.5, "time/run_execution": 1.2},
            {"reward": 0.8, "time/run_execution": 1.5},
        ]
        result = aggregate_metrics(task_metrics, prefix="rollout")
        assert "rollout/reward/mean" in result
        assert "rollout/reward/max" in result
        assert "rollout/reward/min" in result
        assert result["rollout/reward/mean"] == pytest.approx(0.65)

    def test_eval_metrics_unchanged(self):
        task_metrics = [{"accuracy": 0.9}, {"accuracy": 0.7}]
        result = aggregate_eval_metrics(task_metrics, prefix="eval/gsm8k", detailed_stats=False)
        assert result == {"eval/gsm8k/accuracy": pytest.approx(0.8)}
