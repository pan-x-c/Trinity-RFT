"""Rollout coordinator for async batch submission and finalize."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import ray

from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.common.config import Config
from trinity.common.models import InferenceModel
from trinity.common.workflows import Task
from trinity.explorer.scheduler import CompletedTaskResult, Scheduler
from trinity.utils.log import get_logger
from trinity.utils.monitor import gather_eval_metrics, gather_metrics

BatchId = Union[int, str]
BatchType = Literal["train", "eval"]


class BatchLifecycleState(str, Enum):
    """Lifecycle states for one submitted batch."""

    PENDING = "pending"
    RUNNING = "running"
    READY_TO_FINALIZE = "ready_to_finalize"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    ABORTED = "aborted"


class FinalizeReason(str, Enum):
    """Reasons why a batch finalize call returns."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ABORT = "abort"


@dataclass
class BatchState:
    """In-memory state tracked for one train or eval batch."""

    batch_id: BatchId
    batch_type: BatchType
    expected_task_count: int
    completed_task_count: int = 0
    statuses: Dict[Union[int, str], Any] = field(default_factory=dict)
    staged_task_ids: set[int] = field(default_factory=set)
    min_wait_num: Optional[int] = None
    state: BatchLifecycleState = BatchLifecycleState.PENDING
    final_result: Optional[dict] = None
    finalize_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    min_threshold_reached_time: Optional[float] = None

    @property
    def has_partial_results(self) -> bool:
        """Whether at least one task has completed."""

        return self.completed_task_count > 0


class RolloutCoordinator:
    """Own scheduler-side batch state and expose batch-level finalize APIs."""

    def __init__(
        self,
        config: Config,
        rollout_model: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        """Create a coordinator with internally managed scheduler and pipeline."""

        self.logger = get_logger("rollout_coordinator", in_ray_actor=True)
        self.config = config
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models or []
        self.experience_pipeline = None
        self.scheduler: Optional[Scheduler] = None
        self.pending_batches: Dict[BatchId, BatchState] = {}
        self.terminal_batch_results: Dict[BatchId, dict] = {}
        self.event_loop_task: Optional[asyncio.Task] = None
        self.running = False
        self.detailed_stats = getattr(getattr(config, "monitor", None), "detailed_stats", False)

    async def prepare(self) -> None:
        """Initialize the owned pipeline and scheduler."""

        if self.running:
            return
        if self.experience_pipeline is None:
            self.experience_pipeline = self._init_experience_pipeline()
            if self.experience_pipeline is not None:
                await self.experience_pipeline.prepare()
        if self.scheduler is None:
            self.scheduler = self._init_scheduler()
        await self.scheduler.start()
        self.running = True
        self.event_loop_task = asyncio.create_task(self._completed_task_event_loop())

    async def shutdown(self) -> None:
        """Stop background work and close owned dependencies."""

        self.running = False
        if self.event_loop_task is not None:
            self.event_loop_task.cancel()
            try:
                await self.event_loop_task
            except asyncio.CancelledError:
                pass
            self.event_loop_task = None
        if self.scheduler is not None:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline is not None:
            await self.experience_pipeline.close()
            self.experience_pipeline = None

    def _init_experience_pipeline(self):
        """Create the experience pipeline owned by this coordinator actor."""

        if self.config.mode == "bench":
            return None
        return ExperiencePipeline(self.config)

    def _init_scheduler(self) -> Scheduler:
        """Create the scheduler owned by this coordinator."""

        return Scheduler(
            self.config,
            self.rollout_model,
            self.auxiliary_models,
            emit_completed_task_events=True,
        )

    def _require_scheduler(self) -> Scheduler:
        """Return the initialized scheduler."""

        assert self.scheduler is not None, "RolloutCoordinator.prepare() must be called first."
        return self.scheduler

    async def submit_batch(
        self,
        *,
        batch_id: BatchId,
        tasks: list[Task],
        batch_type: BatchType,
        min_wait_num: Optional[int] = None,
    ) -> None:
        """Register a new batch and schedule its tasks."""

        self.terminal_batch_results.pop(batch_id, None)
        existing_state = self.pending_batches.get(batch_id)
        if existing_state is not None and existing_state.state not in {
            BatchLifecycleState.FINALIZED,
            BatchLifecycleState.ABORTED,
        }:
            raise ValueError(f"Batch {batch_id} is already active.")

        batch_state = BatchState(
            batch_id=batch_id,
            batch_type=batch_type,
            expected_task_count=len(tasks),
            min_wait_num=min_wait_num,
        )
        self.pending_batches[batch_id] = batch_state

        if tasks:
            self._require_scheduler().schedule(tasks, batch_id=batch_id)
            batch_state.state = BatchLifecycleState.RUNNING
        else:
            batch_state.state = BatchLifecycleState.READY_TO_FINALIZE

    async def finalize_train_batch(
        self,
        batch_id: int,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        """Finalize one train batch and return aggregated metrics."""

        terminal_result = self.terminal_batch_results.get(batch_id)
        if terminal_result is not None:
            return dict(terminal_result)
        batch_state = self._get_batch_state(batch_id, expected_type="train")
        return await self._finalize_batch(batch_state, timeout=timeout)

    async def finalize_eval_batch(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        """Finalize one eval batch and return aggregated eval metrics."""

        terminal_result = self.terminal_batch_results.get(batch_id)
        if terminal_result is not None:
            return dict(terminal_result)
        scheduler = self._require_scheduler()
        batch_state = self._get_batch_state(batch_id, expected_type="eval")
        async with batch_state.finalize_lock:
            if batch_state.final_result is not None:
                return dict(batch_state.final_result)
            if batch_state.state == BatchLifecycleState.ABORTED:
                batch_state.final_result = self._build_batch_result(
                    batch_state, FinalizeReason.ABORT, {}
                )
                return dict(batch_state.final_result)

            statuses = await scheduler.get_statuses(
                batch_id=batch_id,
                timeout=timeout,
                return_partial_tasks=self._return_partial_tasks(),
            )
            for task_id, status in enumerate(statuses):
                if task_id in batch_state.statuses:
                    continue
                batch_state.statuses[task_id] = status
                batch_state.completed_task_count += 1
            reason = (
                FinalizeReason.COMPLETE
                if batch_state.completed_task_count >= batch_state.expected_task_count
                else FinalizeReason.TIMEOUT
            )
            batch_state.state = BatchLifecycleState.FINALIZED
            batch_state.final_result = self._build_batch_result(batch_state, reason, {})
            self._cache_terminal_batch_result(batch_state)
            return dict(batch_state.final_result)

    async def abort_batch(
        self,
        batch_id: BatchId,
        *,
        reason: str,
        keep_partial_results: bool = False,
    ) -> None:
        """Abort one batch and cleanup its running and staged state."""

        scheduler = self._require_scheduler()
        batch_state = self.pending_batches.get(batch_id)
        if batch_state is None:
            return
        if batch_state.state in {BatchLifecycleState.FINALIZED, BatchLifecycleState.ABORTED}:
            return

        self.logger.warning("Abort batch %s: %s", batch_id, reason)
        await scheduler.abort_batch(
            batch_id,
            return_partial_tasks=keep_partial_results,
            restart_runners=True,
        )
        if self.experience_pipeline is not None:
            await self.experience_pipeline.abort_batch(batch_id)

        batch_state.state = BatchLifecycleState.ABORTED
        batch_state.final_result = self._build_batch_result(batch_state, FinalizeReason.ABORT, {})
        self._cache_terminal_batch_result(batch_state)

    @classmethod
    def get_actor(
        cls, config: Config, models: List, auxiliary_models: List
    ) -> ray.actor.ActorHandle:
        """Init rollout coordinator for the task-event-completion path."""
        return (
            ray.remote(RolloutCoordinator)
            .options(namespace=config.ray_namespace)
            .remote(
                config,
                models,
                auxiliary_models,
            )
        )

    async def _completed_task_event_loop(self) -> None:
        """Consume task completion events emitted by the scheduler."""

        scheduler = self._require_scheduler()
        while self.running:
            try:
                completed_result = await scheduler.wait_completed_task(timeout=0.1)
                if completed_result is None:
                    continue
                if not isinstance(completed_result.task_id, int):
                    self.logger.warning(
                        "Skip completed task event with non-integer task id: %s",
                        completed_result.task_id,
                    )
                    continue
                batch_state = self.pending_batches.get(completed_result.batch_id)
                if batch_state is None:
                    continue
                await self._store_completed_task_result(batch_state, completed_result)
                self._maybe_mark_ready(batch_state)
            except Exception:  # noqa: BLE001
                self.logger.exception("RolloutCoordinator task event loop failed.")

    async def _store_completed_task_result(
        self, batch_state: BatchState, result: CompletedTaskResult
    ) -> None:
        """Persist one completed task into batch-level aggregation state."""

        if result.task_id in batch_state.statuses:
            return
        batch_state.statuses[result.task_id] = result.status
        batch_state.completed_task_count += 1

        if batch_state.batch_type != "train":
            return

        if self.experience_pipeline is not None and result.experience_payloads:
            staged_task_id = int(result.task_id)
            staged = await self.experience_pipeline.stage_task_payloads(
                batch_state.batch_id,
                staged_task_id,
                result.experience_payloads,
            )
            if staged is not None:
                batch_state.staged_task_ids.add(staged_task_id)
            return

        if self.experience_pipeline is not None and result.status.completed_runs > 0:
            batch_state.staged_task_ids.add(int(result.task_id))
            return

    def _get_batch_state(self, batch_id: BatchId, *, expected_type: BatchType) -> BatchState:
        """Return one registered batch and validate its type."""

        batch_state = self.pending_batches.get(batch_id)
        if batch_state is None:
            raise KeyError(f"Batch {batch_id} is not registered.")
        if batch_state.batch_type != expected_type:
            raise ValueError(
                f"Batch {batch_id} is {batch_state.batch_type}, expected {expected_type}."
            )
        return batch_state

    def _get_ready_reason(self, batch_state: BatchState) -> Optional[FinalizeReason]:
        """Check whether a batch is ready to finalize and why."""

        if batch_state.state == BatchLifecycleState.ABORTED:
            return FinalizeReason.ABORT
        if batch_state.completed_task_count >= batch_state.expected_task_count:
            return FinalizeReason.COMPLETE
        if batch_state.batch_type != "train":
            return None
        if batch_state.min_wait_num is None:
            return None
        if batch_state.completed_task_count < batch_state.min_wait_num:
            batch_state.min_threshold_reached_time = None
            return None

        if batch_state.min_threshold_reached_time is None:
            batch_state.min_threshold_reached_time = time.time()

        wait_after_min = getattr(self.config.explorer.over_rollout, "wait_after_min", 0.0)
        if time.time() - batch_state.min_threshold_reached_time >= wait_after_min:
            return FinalizeReason.PARTIAL
        return None

    def _maybe_mark_ready(self, batch_state: BatchState) -> Optional[FinalizeReason]:
        """Transition a batch to ready state when finalize conditions are met."""

        ready_reason = self._get_ready_reason(batch_state)
        if ready_reason is None:
            return None
        if batch_state.state not in {
            BatchLifecycleState.FINALIZED,
            BatchLifecycleState.ABORTED,
            BatchLifecycleState.FINALIZING,
        }:
            batch_state.state = BatchLifecycleState.READY_TO_FINALIZE
        return ready_reason

    async def _wait_for_ready(
        self, batch_state: BatchState, timeout: Optional[float]
    ) -> Optional[FinalizeReason]:
        """Wait until a batch is ready to finalize or the timeout expires."""

        start_time = time.time()
        while True:
            ready_reason = self._maybe_mark_ready(batch_state)
            if ready_reason is not None:
                return ready_reason
            if timeout is not None and (time.time() - start_time) >= timeout:
                return None
            await asyncio.sleep(0.05)

    async def _finalize_batch(self, batch_state: BatchState, *, timeout: Optional[float]) -> dict:
        """Finalize one train batch with idempotent result reuse."""

        async with batch_state.finalize_lock:
            if batch_state.final_result is not None:
                return dict(batch_state.final_result)
            if batch_state.state == BatchLifecycleState.ABORTED:
                batch_state.final_result = self._build_batch_result(
                    batch_state, FinalizeReason.ABORT, {}
                )
                return dict(batch_state.final_result)

            ready_reason = await self._wait_for_ready(batch_state, timeout)
            if ready_reason is None:
                if batch_state.min_wait_num is not None and batch_state.has_partial_results:
                    ready_reason = FinalizeReason.TIMEOUT
                else:
                    raise TimeoutError(f"Timeout waiting for batch {batch_state.batch_id}.")

            batch_state.state = BatchLifecycleState.FINALIZING
            try:
                pipeline_metrics = await self._finalize_train_payloads(batch_state)
                if ready_reason != FinalizeReason.COMPLETE:
                    await self._cleanup_train_batch_runtime(batch_state)
            except Exception:
                batch_state.state = BatchLifecycleState.READY_TO_FINALIZE
                raise

            batch_state.state = BatchLifecycleState.FINALIZED
            batch_state.final_result = self._build_batch_result(
                batch_state, ready_reason, pipeline_metrics
            )
            self._cache_terminal_batch_result(batch_state)
            return dict(batch_state.final_result)

    async def _cleanup_train_batch_runtime(self, batch_state: BatchState) -> None:
        """Drop unfinished train work after a non-complete finalize result."""

        scheduler = self._require_scheduler()
        await scheduler.abort_batch(
            batch_state.batch_id,
            return_partial_tasks=False,
            restart_runners=True,
        )
        if self.experience_pipeline is not None:
            await self.experience_pipeline.abort_batch(batch_state.batch_id)

    def _cache_terminal_batch_result(self, batch_state: BatchState) -> None:
        """Store one terminal result outside the active batch map for idempotent reuse."""

        if batch_state.final_result is None:
            return
        self.terminal_batch_results[batch_state.batch_id] = dict(batch_state.final_result)
        self.pending_batches.pop(batch_state.batch_id, None)

    async def _finalize_train_payloads(self, batch_state: BatchState) -> dict:
        """Flush staged train payloads through the experience pipeline."""

        if self.experience_pipeline is not None and batch_state.staged_task_ids:
            return await self.experience_pipeline.finalize_batch(
                batch_state.batch_id,
                task_ids=sorted(batch_state.staged_task_ids),
            )
        return {}

    def _build_batch_result(
        self,
        batch_state: BatchState,
        reason: FinalizeReason,
        pipeline_metrics: dict,
    ) -> dict:
        """Build the public finalize result returned to Explorer."""

        metrics = dict(pipeline_metrics)
        status_metrics = [
            status.metrics[0] for status in batch_state.statuses.values() if status.metrics
        ]
        if batch_state.batch_type == "train":
            if status_metrics:
                metrics.update(gather_metrics(status_metrics, "rollout"))
            metrics["rollout/finished_task_count"] = float(batch_state.completed_task_count)
        else:
            prefix = self._eval_metric_prefix(batch_state.batch_id)
            if status_metrics:
                metrics.update(
                    gather_eval_metrics(
                        status_metrics,
                        prefix,
                        detailed_stats=self.detailed_stats,
                    )
                )
            metrics[f"{prefix}/finished_task_count"] = float(batch_state.completed_task_count)

        return {
            "batch_id": batch_state.batch_id,
            "batch_type": batch_state.batch_type,
            "finished_task_count": batch_state.completed_task_count,
            "metrics": metrics,
            "finalize_reason": reason.value,
            "finalized": reason != FinalizeReason.ABORT,
        }

    def _eval_metric_prefix(self, batch_id: BatchId) -> str:
        """Return the metric namespace prefix for one eval batch."""

        batch_name = str(batch_id)
        if "/" in batch_name:
            batch_name = batch_name.split("/", 1)[1]
        return f"eval/{batch_name}"

    def _return_partial_tasks(self) -> bool:
        """Return whether scheduler cleanup may emit partial task results."""

        return bool(getattr(self.config.explorer.over_rollout, "return_partial_tasks", False))
