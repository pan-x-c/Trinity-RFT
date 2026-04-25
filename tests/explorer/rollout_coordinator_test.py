"""Unit tests for RolloutCoordinator."""

import asyncio
import unittest
from types import SimpleNamespace

from trinity.explorer.rollout_coordinator import RolloutCoordinator
from trinity.explorer.scheduler import CompletedTaskResult
from trinity.explorer.workflow_runner import Status


class FakePipeline:
    """Minimal in-memory pipeline double for coordinator tests."""

    def __init__(self):
        """Initialize call tracking state."""

        self.stage_calls = []
        self.finalize_calls = []
        self.process_chunk_calls = []
        self.abort_calls = []
        self.prepare_called = False
        self.close_called = False

    async def prepare(self):
        """Record pipeline preparation."""

        self.prepare_called = True

    async def stage_task_payloads(self, batch_id, task_id, exp_chunks):
        """Record task payload staging."""

        self.stage_calls.append((batch_id, task_id, list(exp_chunks)))
        return f"{batch_id}:{task_id}" if exp_chunks else None

    async def finalize_batch(self, batch_id, task_ids=None):
        """Record batch finalization."""

        task_ids = [] if task_ids is None else list(task_ids)
        self.finalize_calls.append((batch_id, task_ids))
        return {"experience_pipeline/experience_count": float(len(task_ids))}

    async def process_serialized_chunks(self, exp_chunks):
        """Record serialized chunk processing."""

        chunks = list(exp_chunks)
        self.process_chunk_calls.append(chunks)
        return {"experience_pipeline/experience_count": float(len(chunks))}

    async def abort_batch(self, batch_id):
        """Record batch abort cleanup."""

        self.abort_calls.append(batch_id)

    async def close(self):
        """Record pipeline closure."""

        self.close_called = True


class FakeScheduler:
    """Minimal scheduler double for coordinator tests."""

    def __init__(self):
        """Initialize scheduler state and recorded calls."""

        self.default_timeout = 1.0
        self.started = False
        self.stopped = False
        self.schedule_calls = []
        self.abort_calls = []
        self.completed_task_results = asyncio.Queue()
        self.batch_results = {}
        self.get_statuses_calls = []

    async def start(self):
        """Mark the scheduler as started."""

        self.started = True

    async def stop(self):
        """Mark the scheduler as stopped."""

        self.stopped = True

    def schedule(self, tasks, batch_id):
        """Record scheduled tasks for one batch."""

        self.schedule_calls.append((batch_id, list(tasks)))

    async def wait_completed_task(self, timeout=None):
        """Return the next queued completed task result."""

        try:
            if timeout is None:
                return await self.completed_task_results.get()
            return await asyncio.wait_for(self.completed_task_results.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def get_statuses(
        self,
        batch_id,
        min_num=None,
        timeout=None,
        clear_timeout_tasks=True,
        return_partial_tasks=False,
    ):
        """Return only preconfigured statuses for one batch."""

        self.get_statuses_calls.append(
            {
                "batch_id": batch_id,
                "min_num": min_num,
                "timeout": timeout,
                "clear_timeout_tasks": clear_timeout_tasks,
                "return_partial_tasks": return_partial_tasks,
            }
        )
        statuses, _ = self.batch_results.pop(batch_id, ([], []))
        return statuses

    async def abort_batch(self, batch_id, return_partial_tasks=False, restart_runners=True):
        """Record one scheduler abort request."""

        self.abort_calls.append(
            {
                "batch_id": batch_id,
                "return_partial_tasks": return_partial_tasks,
                "restart_runners": restart_runners,
            }
        )

    def emit_completed_task(self, batch_id, task_id, result):
        """Queue one completed task event and result."""

        self.completed_task_results.put_nowait(result)


def _build_config(wait_after_min=0.0, return_partial_tasks=True, detailed_stats=False):
    return SimpleNamespace(
        explorer=SimpleNamespace(
            over_rollout=SimpleNamespace(
                wait_after_min=wait_after_min,
                return_partial_tasks=return_partial_tasks,
            )
        ),
        monitor=SimpleNamespace(detailed_stats=detailed_stats),
    )


def _build_status(metric_value):
    return Status(
        completed_runs=1,
        total_runs=1,
        metrics=[{"run_metrics": float(metric_value)}],
    )


class CoordinatorHarness(RolloutCoordinator):
    """Coordinator subclass that injects fake owned dependencies."""

    def __init__(self, config, rollout_model, auxiliary_models=None, *, pipeline, scheduler):
        """Store doubles and delegate the main initialization to the parent."""

        self._test_pipeline = pipeline
        self._test_scheduler = scheduler
        super().__init__(config, rollout_model, auxiliary_models)

    def _init_experience_pipeline(self):
        """Return the injected fake pipeline."""

        return self._test_pipeline

    def _init_scheduler(self):
        """Return the injected fake scheduler."""

        return self._test_scheduler


class TestRolloutCoordinator(unittest.IsolatedAsyncioTestCase):
    """Focused behavioral tests for the first coordinator implementation."""

    async def asyncSetUp(self):
        """Create one coordinator wired to fake owned dependencies."""

        self.scheduler = FakeScheduler()
        self.pipeline = FakePipeline()
        self.coordinator = CoordinatorHarness(
            _build_config(),
            rollout_model=[],
            pipeline=self.pipeline,
            scheduler=self.scheduler,
        )
        await self.coordinator.prepare()

    async def asyncTearDown(self):
        """Shutdown the coordinator after each test."""

        await self.coordinator.shutdown()

    async def test_finalize_train_batch_tracks_scheduler_events_and_is_idempotent(self):
        """Train finalize should consume task events once and reuse the final result."""

        await self.coordinator.submit_batch(
            batch_id=1,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
        )

        self.scheduler.emit_completed_task(
            1,
            0,
            CompletedTaskResult(
                batch_id=1,
                task_id=0,
                status=_build_status(10.0),
                experience_payloads=[b"payload-0"],
            ),
        )
        self.scheduler.emit_completed_task(
            1,
            1,
            CompletedTaskResult(
                batch_id=1,
                task_id=1,
                status=_build_status(20.0),
                experience_payloads=[b"payload-1"],
            ),
        )

        result = await self.coordinator.finalize_train_batch(1, timeout=1.0)
        repeated = await self.coordinator.finalize_train_batch(1, timeout=1.0)

        self.assertEqual(result["finalize_reason"], "complete")
        self.assertEqual(result["finished_task_count"], 2)
        self.assertEqual(result["metrics"]["rollout/run_metrics/mean"], 15.0)
        self.assertEqual(result["metrics"]["experience_pipeline/experience_count"], 2.0)
        self.assertTrue(self.pipeline.prepare_called)
        self.assertEqual(len(self.pipeline.stage_calls), 2)
        self.assertEqual(self.pipeline.finalize_calls, [(1, [0, 1])])
        self.assertEqual(result, repeated)
        self.assertNotIn(1, self.coordinator.pending_batches)
        self.assertEqual(
            self.coordinator.terminal_batch_results[1]["finalize_reason"],
            "complete",
        )

    async def test_finalize_train_batch_supports_partial_finalize(self):
        """Train finalize should allow partial completion when policy permits it."""

        await self.coordinator.submit_batch(
            batch_id=2,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
            min_wait_num=1,
            allow_partial_finalize=True,
        )

        self.scheduler.emit_completed_task(
            2,
            0,
            CompletedTaskResult(
                batch_id=2,
                task_id=0,
                status=_build_status(7.0),
                experience_payloads=[b"payload-0"],
            ),
        )

        result = await self.coordinator.finalize_train_batch(2, timeout=1.0)

        self.assertEqual(result["finalize_reason"], "partial")
        self.assertEqual(result["finished_task_count"], 1)
        self.assertEqual(self.pipeline.finalize_calls[-1], (2, [0]))
        self.assertEqual(self.scheduler.abort_calls[-1]["batch_id"], 2)
        self.assertEqual(self.pipeline.abort_calls[-1], 2)
        self.assertNotIn(2, self.coordinator.pending_batches)

    async def test_finalize_eval_batch_aggregates_eval_metrics(self):
        """Eval finalize should aggregate scheduler results without pipeline writes."""

        batch_id = "3/eval_set"
        await self.coordinator.submit_batch(
            batch_id=batch_id,
            tasks=[SimpleNamespace(is_eval=True), SimpleNamespace(is_eval=True)],
            batch_type="eval",
        )
        self.scheduler.batch_results[batch_id] = (
            [_build_status(3.0), _build_status(5.0)],
            [],
        )

        result = await self.coordinator.finalize_eval_batch(batch_id, timeout=1.0)

        self.assertEqual(result["finalize_reason"], "complete")
        self.assertEqual(result["finished_task_count"], 2)
        self.assertEqual(result["metrics"]["eval/eval_set/run_metrics"], 4.0)
        self.assertEqual(self.pipeline.finalize_calls, [])
        self.assertEqual(self.scheduler.get_statuses_calls[0]["batch_id"], batch_id)
        self.assertNotIn(batch_id, self.coordinator.pending_batches)

    async def test_abort_batch_marks_batch_aborted_and_is_visible_to_finalize(self):
        """Abort should short-circuit later finalize calls."""

        await self.coordinator.submit_batch(
            batch_id=4,
            tasks=[SimpleNamespace(is_eval=False), SimpleNamespace(is_eval=False)],
            batch_type="train",
        )

        await self.coordinator.abort_batch(4, reason="shutdown")
        result = await self.coordinator.finalize_train_batch(4, timeout=0.1)

        self.assertEqual(result["finalize_reason"], "abort")
        self.assertFalse(result["finalized"])
        self.assertEqual(self.scheduler.abort_calls[0]["batch_id"], 4)
        self.assertEqual(self.pipeline.abort_calls, [4])
        self.assertNotIn(4, self.coordinator.pending_batches)

    async def test_shutdown_closes_internal_dependencies(self):
        """Shutdown should close both owned scheduler and owned pipeline."""

        await self.coordinator.shutdown()

        self.assertTrue(self.scheduler.stopped)
        self.assertTrue(self.pipeline.close_called)
