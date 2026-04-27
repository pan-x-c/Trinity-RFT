from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import ray
from ray.exceptions import GetTimeoutError, RayTaskError

from trinity.buffer.pipelines.task_pipeline import check_and_run_task_pipeline
from trinity.common.config import Config, load_config
from trinity.explorer.explorer import Explorer
from trinity.perf import (
    ResourceSampler,
    TensorBoardScalarReader,
    build_global_metrics,
    build_resource_timeline_payload,
    collect_step_metrics,
)
from trinity.utils.plugin_loader import load_plugins


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Explorer performance collection.")
    parser.add_argument("--config", required=True, help="Path to the Trinity explorer config.")
    parser.add_argument("--output-path", required=True, help="Path to the output JSON file.")
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=5.0,
        help="Resource sampling interval in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for prepare, sync and explore calls.",
    )
    return parser.parse_args()


def validate_config(config: Config) -> None:
    """Validate perf-specific config constraints."""
    if config.mode != "explore":
        raise ValueError(f"Explorer perf requires mode 'explore', got '{config.mode}'.")
    if config.monitor.monitor_type != "tensorboard":
        raise ValueError(
            "Explorer perf requires monitor.monitor_type='tensorboard' so step metrics can "
            "be read from local event files."
        )


def build_output_payload(
    *,
    config: Optional[Config],
    config_path: str,
    output_path: str,
    monitor_interval: float,
    startup_time_sec: Optional[float],
    run_time_sec: Optional[float],
    total_time_sec: Optional[float],
    resource_payload: dict[str, Any],
    step_metrics: list[dict[str, Any]],
    success: bool,
    error: Optional[str],
) -> dict[str, Any]:
    """Assemble the final JSON payload."""
    artifacts = {}
    explorer_name = None
    if config is not None:
        explorer_name = config.explorer.name
        artifacts = {
            "checkpoint_job_dir": config.checkpoint_job_dir,
            "tensorboard_dir": os.path.join(
                config.monitor.cache_dir, "tensorboard", config.explorer.name
            ),
            "log_dir": config.log.save_dir,
            "output_json": output_path,
        }

    return {
        "run_meta": {
            "config_path": config_path,
            "explorer_name": explorer_name,
            "monitor_interval_sec": monitor_interval,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "generated_at": time.time(),
        },
        "timing": {
            "startup_time_sec": startup_time_sec,
            "run_time_sec": run_time_sec,
            "total_time_sec": total_time_sec,
        },
        **resource_payload,
        "step_metrics": step_metrics,
        "global_metrics": build_global_metrics(step_metrics),
        "artifacts": artifacts,
        "status": {
            "success": success,
            "error": error,
            "gpu_metrics_available": bool(resource_payload.get("resource_timeline")),
            "tensorboard_metrics_available": bool(step_metrics),
        },
    }


def write_output(output_path: str, payload: dict[str, Any]) -> None:
    """Write the final payload to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_explorer_perf(args: argparse.Namespace) -> dict[str, Any]:
    """Run Explorer perf collection and return the result payload."""
    load_plugins()
    config: Optional[Config] = None
    explorer_actor = None
    sampler: Optional[ResourceSampler] = None
    success = False
    error: Optional[str] = None
    startup_time_sec: Optional[float] = None
    run_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    resource_payload: dict[str, Any] = {"resource_timeline": [], "chart_series": {}}
    step_metrics: list[dict[str, Any]] = []
    startup_started_at: Optional[float] = None
    run_started_at: Optional[float] = None

    try:
        config = load_config(args.config)
        validate_config(config)
        config.check_and_update()

        ray.init(
            address=config.cluster.ray_address,
            ignore_reinit_error=True,
            namespace=config.ray_namespace,
            runtime_env={"env_vars": config.get_envs()},
        )
        check_and_run_task_pipeline(config)

        sampler = ResourceSampler(interval_seconds=args.monitor_interval)
        sampler.start()

        startup_started_at = time.perf_counter()
        explorer_actor = Explorer.get_actor(config)
        ray.get(explorer_actor.prepare.remote(), timeout=args.timeout)
        startup_time_sec = time.perf_counter() - startup_started_at

        run_started_at = time.perf_counter()
        ray.get(explorer_actor.sync_weight.remote(), timeout=args.timeout)
        ray.get(explorer_actor.explore.remote(), timeout=args.timeout)
        run_time_sec = time.perf_counter() - run_started_at
        total_time_sec = time.perf_counter() - startup_started_at
        success = True
    except (RuntimeError, ValueError, TimeoutError, GetTimeoutError, RayTaskError):
        error = traceback.format_exc()
        if startup_started_at is not None and startup_time_sec is None and run_started_at is None:
            startup_time_sec = time.perf_counter() - startup_started_at
        if run_started_at is not None and run_time_sec is None:
            run_time_sec = time.perf_counter() - run_started_at
        if startup_started_at is not None and total_time_sec is None:
            total_time_sec = time.perf_counter() - startup_started_at
    finally:
        collected_samples = sampler.stop() if sampler is not None else []
        resource_payload = build_resource_timeline_payload(collected_samples)

        if explorer_actor is not None and ray.is_initialized():
            try:
                ray.get(explorer_actor.shutdown.remote(), timeout=args.timeout)
            except (RuntimeError, TimeoutError, GetTimeoutError, RayTaskError):
                if error is None:
                    error = traceback.format_exc()

        if config is not None:
            tensorboard_dir = os.path.join(
                config.monitor.cache_dir, "tensorboard", config.explorer.name
            )
            if os.path.isdir(tensorboard_dir):
                scalar_reader = TensorBoardScalarReader(tensorboard_dir)
                step_metrics = collect_step_metrics(scalar_reader.metrics)

        if ray.is_initialized():
            ray.shutdown()

    return build_output_payload(
        config=config,
        config_path=args.config,
        output_path=args.output_path,
        monitor_interval=args.monitor_interval,
        startup_time_sec=startup_time_sec,
        run_time_sec=run_time_sec,
        total_time_sec=total_time_sec,
        resource_payload=resource_payload,
        step_metrics=step_metrics,
        success=success,
        error=error,
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    payload = run_explorer_perf(args)
    write_output(args.output_path, payload)
    return 0 if payload["status"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
