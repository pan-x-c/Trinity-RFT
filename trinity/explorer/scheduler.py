"""Scheduler for rollout tasks."""

import asyncio
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import traceback

import ray

from trinity.common.models import InferenceModel
from trinity.common.config import Config
from trinity.common.workflows import Task
from trinity.explorer.workflow_runner import WorkflowRunner, Status
from trinity.utils.log import get_logger



class RunnerWrapper:

    def __init__(self, runner: WorkflowRunner, runner_id: int):
        self.logger = get_logger(__name__)
        self.runner = runner
        self.runner_id = runner_id
        self.is_busy = False
        self.current_task: Task = None

    async def run_with_retry(self, task: Task, retry_times: int) -> Tuple[Status, int]:
        """
        Returns:
            `Status`: The return status of the task.
            `int`: The runner_id of current runner.
        """
        last_exception_msg = None
        self.is_busy = True
        self.current_task = task
        start_time = time.time()
        try:
            for attempt in range(retry_times + 1):
                try:
                    status = await self.runner.run.remote(task)
                    if status.ok:
                        break
                    else:
                        self.logger.error(status.message)
                except Exception:
                    last_exception_msg = traceback.format_exception()
                    self.logger.warning(
                        f"Task execution attempt {attempt + 1} failed:\n{last_exception_msg}"
                    )
                    status = Status(ok=False, metric=dict(), message=last_exception_msg)
        finally:
            end_time = time.time()
            status.metric["task_run_time"] = end_time - start_time
            self.is_busy = False
            self.current_task = None
        return status, self.runner_id


class Scheduler:
    """Scheduler for rollout tasks."""

    def __init__(
        self,
        config: Config,
        rollout_model: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        self.logger = get_logger(__name__)
        self.config = config
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models or []
        self.namespace = ray.get_runtime_context().namespace
        self.timeout = config.explorer.max_timeout
        self.max_retry_times = config.explorer.max_retry_times
        self.running = False

        self.runner_num = len(rollout_model) * config.explorer.runner_per_model
        self.runners: Dict[int, RunnerWrapper] = dict()
        self.idle_runners = set()
        self.busy_runners = dict()

        self.pending_tasks: Dict[int, deque] = defaultdict(deque)  # step -> tasks
        self.running_tasks: Dict[int, set[asyncio.Future]] = defaultdict(set)  # step -> futures
        self.completed_tasks: Dict[int, deque[Status]] = defaultdict(deque)  # step -> results

        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False

        self.total_scheduled = 0
        self.total_completed = 0
        for i in range(self.runner_num):
            self._create_runner(i)

    async def _create_runner(
        self,
        runner_id: int,
    ) -> None:
        runner = RunnerWrapper(
            runner=(
                ray.remote(WorkflowRunner)
                .options(
                    namespace=self.namespace,
                    scheduling_strategy="SPREAD",
                )
                .remote(
                    self.config,
                    self.rollout_model[runner_id % len(self.rollout_model)],
                    [
                        self.auxiliary_models[j][runner_id % len(self.auxiliary_models[j])]
                        for j in range(len(self.auxiliary_models))
                    ],
                )
            ),
            runner_id=runner_id,
        )
        self.runners[runner_id] = runner
        self.idle_runners.add(runner_id)

    def _restart_runner(self, runner_id: int):
        """Restart a runner."""
        try:
            ray.kill(self.runners[runner_id])
        except:
            pass
        
        self.create_runner(runner_id)


    async def _scheduler_loop(self) -> None:
        self.logger.info("Scheduler loop started.")
        while self.running:
            try:
                await self._schedule_pending_tasks()
                await self._check_completed_tasks()
                await asyncio.sleep(0.01)
            except Exception:
                self.logger.error(f"Error in scheduler loop:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)
        self.logger.info("Scheduler loop stopped.")

    async def _schedule_pending_tasks(self) -> None:
        if not self.idle_runners:
            return

        for step in sorted(self.pending_tasks.keys()):
            task_queue = self.pending_tasks[step]

            while task_queue and self.idle_runners:
                task = task_queue.pop()
                runner_id = self.idle_runners.pop()
                self.busy_runners[runner_id] = (task, step)
                self.running_tasks[step].add(
                    asyncio.create_task(self.runners[runner_id].run_with_retry(task))
                )

            if not task_queue:
                del self.pending_tasks[step]

    async def _check_completed_tasks(self) -> None:
        for step in list(self.running_tasks.keys()):
            futures = self.running_tasks[step]

            for future in list(futures):
                if future.done():
                    futures.remove(future)
                    try:
                        task_result, runner_id = await future
                        self.completed_tasks[step].appendleft(task_result)
                        self.busy_runners.pop(runner_id)
                        self.idle_runners.add(runner_id)

                        self.logger.debug(
                            f"Task completed (step {step}), success: {task_result.success}"
                        )

                    except Exception as e:
                        self.logger.error(f"Error getting task result: {e}")

            if not futures:
                del self.running_tasks[step]

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await asyncio.gather([self._create_runner(i) for i in range(self.runner_num)])
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        all_running_futures = []
        for futures in self.running_tasks.values():
            all_running_futures.extend(futures)

        if all_running_futures:
            self.logger.info(f"Waiting for {len(all_running_futures)} running tasks to complete...")
            await asyncio.gather(*all_running_futures, return_exceptions=True)

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Scheduler stopped")

    def schedule(self, tasks: List[Task], step: int) -> None:
        """Schedule the provided tasks.

        Args:
            tasks (`List[Task]`): The tasks to schedule.
            step (`int`): The step number of provided tasks.
        """
        if not tasks:
            return
        for task in tasks:
            self.pending_tasks[step].appendleft(task)


    async def get_results(
        self, step: int, min_num: Optional[int] = None, timeout: Optional[float] = None
    ) -> List[Dict]:
        """Get the result of tasks at the specific step.

        Args:
            step (`int`): Only wait for tasks at this step.
            min_num (`int`): The minimum number of tasks to wait for. If `None`, wait for all tasks at `step`.
            timeout (`float`): The timeout for waiting for tasks to finish. If `None`, wait for default timeout.
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        if min_num is None:
            min_num = len(self.pending_tasks[step]) + len(self.running_tasks[step]) + len(self.completed_tasks[step])
        self.logger.debug(f"Waiting for {min_num} tasks to complete...")

        while time.time() - start_time < timeout:
            completed_count = len(self.completed_tasks[step])
            if completed_count >= min_num:
                break
            await asyncio.sleep(0.1)

        results = []
        for _ in range(min_num):
            if len(self.completed_tasks[step]) > 0:
                results.append(self.completed_tasks[step].pop())

        if not self.completed_tasks[step]:
            del self.completed_tasks[step]

        completed_count = len(results)
        if completed_count < min_num:
            self.logger.warning(
                f"Timeout reached, only {completed_count}/{min_num} tasks completed"
            )

        return results
