# -*- coding: utf-8 -*-
"""The Workflow Runner Module."""

import asyncio
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

from trinity.buffer import get_buffer_reader, get_buffer_writer
from trinity.common.config import Config, StorageConfig
from trinity.common.constants import LOG_DIR_ENV_VAR, LOG_LEVEL_ENV_VAR
from trinity.common.experience import Experience
from trinity.common.models.allocator import Allocator
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import RepeatableWorkflow, Status, Task, Workflow
from trinity.utils.log import get_logger


class WorkflowRunner:
    """A Ray remote actor that runs workflows and returns execution statuses.

    Experience payloads are not returned through the runner. The rollout model
    owns experience capture through its recording/history path, and the rollout
    coordinator drains those model-side stores at step finalization.
    """

    def __init__(
        self,
        config: Config,
        rollout_model_id: int,
        auxiliary_model_ids: Optional[List[int]] = None,
        runner_id: Optional[int] = None,
    ) -> None:
        self.name = f"{config.explorer.name}_runner_{runner_id}"
        self.logger = get_logger(self.name, in_ray_actor=True)
        self.config = config
        allocator = Allocator(config.explorer)
        self.model_wrapper: ModelWrapper = allocator.get_model(
            config.explorer.rollout_model, "rollout", rollout_model_id
        )
        self.auxiliary_model_wrappers: List[ModelWrapper] = [
            allocator.get_model(
                config.explorer.auxiliary_models[index], f"auxiliary_{index}", auxiliary_model_id
            )
            for index, auxiliary_model_id in enumerate(auxiliary_model_ids or [])
        ]
        self.workflow_instance: Workflow = None
        self.rollout_model_id = rollout_model_id
        self.runner_id = runner_id
        self.runner_state = {
            "workflow_id": None,
            "model_version": None,
            "begin_time": 0,
            "terminate_time": 0,
        }
        self.concurrent_mode = config.explorer.concurrent_mode
        if self.concurrent_mode == "sequential":
            self.concurrent_run_fn = self._sequential_run
        elif self.concurrent_mode == "asynchronous":
            self.concurrent_run_fn = self._asynchronous_run
        elif self.concurrent_mode == "multi-threading":
            self.concurrent_run_fn = self._multi_threading_run
        else:
            self.logger.warning(
                f"Unknown concurrent_mode {self.concurrent_mode}, defaulting to sequential."
            )
            self.concurrent_run_fn = self._sequential_run
        self.logger.info(
            f"WorkflowRunner [{self.name}]({self.concurrent_mode}) initialized:\n"
            f"  > rollout model: {self.config.explorer.rollout_model.model_path}\n"
            f"  > auxiliary models: {[aux_model_config.model_path for aux_model_config in self.config.explorer.auxiliary_models]}"
        )

    async def prepare(self) -> None:
        """Prepare the runner."""
        await asyncio.gather(
            self.model_wrapper.prepare(),
            *(aux_model.prepare() for aux_model in self.auxiliary_model_wrappers),
        )
        self.logger.info(f"WorkflowRunner [{self.name}] is prepared and ready to run tasks.")

    def is_alive(self):
        return True

    def _build_record_key(self, task: Task, run_index: int) -> str:
        return f"{task.batch_id}/{task.task_id}/{run_index}"

    def _set_record_key(self, model_wrapper: ModelWrapper, record_key: Optional[str]) -> None:
        if record_key is not None:
            model_wrapper.set_api_key(record_key)

    def _create_workflow_instance(self, task: Task, record_key: Optional[str] = None) -> Workflow:
        if task.workflow is None:
            raise ValueError("Workflow is not set in the task.")
        self._set_record_key(self.model_wrapper, record_key)
        if (
            self.workflow_instance is None
            or not self.workflow_instance.__class__ == task.workflow
            or not self.workflow_instance.resettable
        ):
            # Pass ModelWrapper directly; Workflow.__init__ will get OpenAI clients automatically
            self.workflow_instance = task.to_workflow(
                self.model_wrapper,
                self.auxiliary_model_wrappers,
            )
        else:
            self.workflow_instance.reset(task)
        return self.workflow_instance

    async def _run_workflow(self, workflow_instance: Workflow) -> Status:
        status = await workflow_instance.execute()
        if not isinstance(status, Status):
            raise TypeError(
                f"{workflow_instance.__class__.__name__}.execute must return Status, "
                f"got {type(status).__name__}."
            )
        return status

    def _create_isolated_workflow_instance(
        self, task: Task, record_key: Optional[str] = None
    ) -> Tuple[Workflow, ModelWrapper]:
        model_wrapper = (
            self.model_wrapper.clone_with_isolated_history()
            if self.config.explorer.rollout_model.enable_history
            else self.model_wrapper
        )
        self._set_record_key(model_wrapper, record_key)
        wf = task.to_workflow(
            model_wrapper,
            self.auxiliary_model_wrappers,
        )
        return wf, model_wrapper

    def _build_status(
        self,
        total_runs: int,
        metrics: List[Dict[str, float]],
        successful_run_ids: List[int],
        first_error: Optional[str] = None,
    ) -> Status:
        completed_runs = len(successful_run_ids)
        if first_error is None:
            message = None
        elif completed_runs > 0:
            message = (
                f"{completed_runs}/{total_runs} runs completed successfully. "
                f"First error: {first_error}"
            )
        else:
            message = first_error

        return Status(
            completed_runs=completed_runs,
            total_runs=total_runs,
            metrics=list(metrics),
            successful_run_ids=list(successful_run_ids),
            message=message,
        )

    def _aggregate_run_results(
        self,
        total_runs: int,
        results: List[Status],
    ) -> Status:
        run_metrics = []
        successful_run_ids = []
        first_error = None

        for status in results:
            if status.ok:
                run_metrics.extend(status.metrics)
                successful_run_ids.extend(status.successful_run_ids)
                continue
            if first_error is None:
                first_error = status.message

        return self._build_status(
            total_runs=total_runs,
            metrics=run_metrics,
            successful_run_ids=successful_run_ids,
            first_error=first_error,
        )

    async def _run_parallel_runs(
        self,
        task: Task,
        repeat_times: int,
        run_id_base: int,
        collect_partial_runs: bool = True,
        use_threads: bool = False,
    ) -> Status:
        async def run_single(i: int) -> Status:
            run_index = run_id_base + i
            record_key = self._build_record_key(task, run_index)
            workflow, model_wrapper = self._create_isolated_workflow_instance(task, record_key)
            return await self._execute_single_run(
                workflow,
                task,
                i,
                run_id_base,
                model_wrapper=model_wrapper,
                record_key=record_key,
            )

        if collect_partial_runs:
            if use_threads:
                results = await asyncio.gather(
                    *(
                        asyncio.to_thread(lambda idx=i: asyncio.run(run_single(idx)))  # type: ignore[misc]
                        for i in range(repeat_times)
                    )
                )
            else:
                results = await asyncio.gather(*(run_single(i) for i in range(repeat_times)))
            return self._aggregate_run_results(repeat_times, results)

        future_to_run_index = {}
        for i in range(repeat_times):
            if use_threads:
                future = asyncio.create_task(
                    asyncio.to_thread(lambda idx=i: asyncio.run(run_single(idx)))  # type: ignore[misc]
                )
            else:
                future = asyncio.create_task(run_single(i))
            future_to_run_index[future] = i

        results = []
        while future_to_run_index:
            done, pending = await asyncio.wait(
                future_to_run_index.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            should_stop = False
            for future in done:
                future_to_run_index.pop(future)
                result = future.result()
                results.append(result)
                if not result.ok:
                    should_stop = True
            if should_stop:
                for future in pending:
                    future.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                break

        return self._aggregate_run_results(repeat_times, results)

    async def _execute_single_run(
        self,
        workflow: Workflow,
        task: Task,
        run_index: int,
        run_id_base: int,
        model_wrapper: Optional[ModelWrapper] = None,
        record_key: Optional[str] = None,
    ) -> Status:
        st = time.time()
        model_wrapper = model_wrapper or self.model_wrapper
        self._set_record_key(model_wrapper, record_key)
        await model_wrapper.clean_workflow_state()
        run_id = run_id_base + run_index
        workflow.set_execution_context(run_id=run_id)
        self.runner_state["workflow_id"] = self._build_record_key(task, run_id)
        self.runner_state["terminate_time"] = None
        self.runner_state["begin_time"] = st
        try:
            status = await self._run_workflow(workflow)
            et = time.time()
            self.runner_state["terminate_time"] = et
            metrics = [dict(metric) for metric in status.metrics]
            if not metrics:
                metrics = [{}]
            for metric in metrics:
                metric["time/run_execution"] = et - st
            successful_run_ids = status.successful_run_ids or (
                [run_id] if status.completed_runs > 0 else []
            )
            status = Status(
                completed_runs=len(successful_run_ids),
                total_runs=status.total_runs,
                metrics=metrics,
                successful_run_ids=successful_run_ids,
                message=status.message,
            )
            return status
        except Exception as exc:
            self.runner_state["terminate_time"] = time.time()
            error_trace_back = traceback.format_exc()
            self.logger.error(
                "WorkflowRunner single run error: " f"{exc}\nTraceback:\n{error_trace_back}"
            )
            return Status(
                completed_runs=0,
                total_runs=1,
                metrics=[],
                message=error_trace_back.rstrip(),
            )

    async def _run_task(
        self,
        task: Task,
        repeat_times: int,
        run_id_base: int,
        collect_partial_runs: bool = True,
    ) -> Status:
        """Init workflow from the task and run it."""
        if issubclass(task.workflow, RepeatableWorkflow):
            record_key = self._build_record_key(task, run_id_base)
            workflow_instance = self._create_workflow_instance(task, record_key=record_key)
            workflow_instance.set_execution_context(repeat_times, run_id_base)
            st = time.time()
            self._set_record_key(self.model_wrapper, record_key)
            await self.model_wrapper.clean_workflow_state()
            self.runner_state["workflow_id"] = record_key
            self.runner_state["terminate_time"] = None
            self.runner_state["begin_time"] = st
            status = await self._run_workflow(workflow_instance)
            et = time.time()
            self.runner_state["terminate_time"] = et
            run_metrics = [dict(metric) for metric in status.metrics]
            for metric in run_metrics:
                metric["time/run_execution"] = et - st
            successful_run_ids = status.successful_run_ids or list(
                range(run_id_base, run_id_base + status.completed_runs)
            )
            return self._build_status(
                total_runs=repeat_times,
                metrics=run_metrics,
                successful_run_ids=successful_run_ids,
                first_error=status.message,
            )
        else:
            return await self.concurrent_run_fn(
                task,
                repeat_times,
                run_id_base,
                collect_partial_runs=collect_partial_runs,
            )

    async def _sequential_run(
        self,
        task: Task,
        repeat_times: int,
        run_id_base: int,
        collect_partial_runs: bool = True,
    ) -> Status:
        results = []
        for i in range(repeat_times):
            run_index = run_id_base + i
            record_key = self._build_record_key(task, run_index)
            workflow = self._create_workflow_instance(task, record_key=record_key)
            result = await self._execute_single_run(
                workflow,
                task,
                i,
                run_id_base,
                record_key=record_key,
            )
            results.append(result)
            if collect_partial_runs:
                continue
            if result.ok:
                continue
            break
        return self._aggregate_run_results(repeat_times, results)

    async def _asynchronous_run(
        self,
        task: Task,
        repeat_times: int,
        run_id_base: int,
        collect_partial_runs: bool = True,
    ) -> Status:
        return await self._run_parallel_runs(
            task,
            repeat_times,
            run_id_base,
            collect_partial_runs=collect_partial_runs,
        )

    async def _multi_threading_run(
        self,
        task: Task,
        repeat_times: int,
        run_id_base: int,
        collect_partial_runs: bool = True,
    ) -> Status:
        return await self._run_parallel_runs(
            task,
            repeat_times,
            run_id_base,
            collect_partial_runs=collect_partial_runs,
            use_threads=True,
        )

    async def get_runner_state(self) -> Dict:
        """Get the runner state."""
        runner_state = self.runner_state.copy()
        runner_state.update(await self.model_wrapper.get_workflow_state())
        return runner_state

    async def run_task(
        self,
        task: Task,
        batch_id: str,
        repeat_times: int = 1,
        run_id_base: int = 0,
        collect_partial_runs: bool = True,
    ) -> Status:
        """Run the task and return its execution status."""
        st = time.time()
        try:
            model_version = await self.model_wrapper.model_version_async
            self.runner_state["model_version"] = model_version
            self.logger.info(
                f"Starting task: step={batch_id}, model_version={model_version}, repeat_times={repeat_times}, run_id_base={run_id_base}"
            )
            status = await self._run_task(
                task,
                repeat_times,
                run_id_base,
                collect_partial_runs=collect_partial_runs,
            )
            return status

        except Exception as e:
            error_trace_back = traceback.format_exc()
            self.logger.error(f"WorkflowRunner run task error: {e}\nTraceback:\n{error_trace_back}")
            return Status(
                completed_runs=0,
                total_runs=repeat_times,
                metrics=[{"time/run_execution": time.time() - st}],
                message=error_trace_back.rstrip(),
            )


class DebugWorkflowRunner(WorkflowRunner):
    """A WorkflowRunner for debugging."""

    def __init__(
        self,
        config: Config,
        output_dir: str = "debug_output",
        enable_profiling: bool = False,
        disable_overwrite: bool = False,
    ) -> None:
        if disable_overwrite:
            # if output dir is not empty, change to a new dir with datetime suffix
            if os.path.isdir(output_dir) and os.listdir(output_dir):
                suffix = time.strftime("%Y%m%d%H%M%S", time.localtime())
                output_dir = f"{output_dir}_{suffix}"
        os.environ[LOG_DIR_ENV_VAR] = os.path.join(output_dir, "log")
        os.environ[LOG_LEVEL_ENV_VAR] = "DEBUG"
        super().__init__(
            config=config,
            rollout_model_id=0,
            auxiliary_model_ids=[0] * len(config.explorer.auxiliary_models),
            runner_id=0,
        )
        self.taskset = get_buffer_reader(config.buffer.explorer_input.tasksets[0])
        self.output_dir = output_dir
        self.enable_profiling = enable_profiling
        self.logger.info(f"Debug output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_profiling_file = os.path.join(
            self.output_dir,
            "profiling.html",
        )
        self.output_sqlite_file = "sqlite:///" + os.path.join(
            self.output_dir,
            "experiences.db",
        )
        self.sqlite_writer = get_buffer_writer(
            StorageConfig(
                name="debug_buffer",
                schema_type="experience",
                path=self.output_sqlite_file,
                storage_type="sql",
                batch_size=1,
                wrap_in_ray=False,
            )
        )

    async def debug(self) -> None:
        """Run the debug workflow."""
        tasks = await self.taskset.read(batch_size=1)
        task = tasks[0]
        self.logger.info(f"Start debugging task:\n{task.raw_task}")
        if not self.enable_profiling:
            status = await self.run_task(task=task, batch_id="debug", repeat_times=1, run_id_base=0)
        else:
            from viztracer import VizTracer

            with VizTracer(output_file=self.output_profiling_file):
                status = await self.run_task(
                    task=task, batch_id="debug", repeat_times=1, run_id_base=0
                )
        experiences = []
        if self.config.explorer.rollout_model.enable_history:
            try:
                payload = await self.model_wrapper.drain_experience_records_bytes_async("debug")
                experiences = Experience.deserialize_many(payload) if payload else []
            except Exception:
                experiences = []
        if not status.ok and not experiences:
            try:
                experiences = self.model_wrapper.extract_experience_from_history()
                self.logger.info(
                    f"Debugging failed, extracting {len(experiences)} experiences from history."
                )
            except Exception:
                experiences = []
        await self.sqlite_writer.write(experiences)
        if status.ok:
            print(f"Task {task.task_id} completed successfully with metrics:\n{status.metrics}")
        else:
            self.logger.error(f"Task {task.task_id} failed with message: {status.message}")
        self.logger.info("Debugging completed.")
