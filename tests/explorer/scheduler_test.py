import time
import unittest
from typing import List, Tuple

import ray
import torch

from tests.tools import get_template_config
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.explorer.scheduler import Scheduler


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    def __init__(self, model, task, auxiliary_models):
        super().__init__(model, task, auxiliary_models)
        self.error_type = task.raw_task.get("error_type", "")
        self.seconds = None
        if "timeout" in self.error_type:
            self.seconds = int(self.error_type.split("_")[-1])

    def run(self) -> List[Experience]:
        if "timeout" in self.error_type:
            time.sleep(self.seconds)
        elif self.error_type == "exception":
            raise ValueError("Exception occurred")
        elif self.error_type == "exit":
            exit(1)
        elif self.error_type == "auxiliary_models":
            assert self.auxiliary_models is not None and len(self.auxiliary_models) == 2
        return [Experience(tokens=torch.zeros(5), prompt_length=2, prompt_text=self.error_type)]


@ray.remote
class DummyModel(InferenceModel):
    def sync_model(self, model_version, update_weight_args_list):
        return True

    def get_model_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        update_with_checkpoint: bool = True,
    ) -> None:
        pass


@ray.remote
class DummyAuxiliaryModel(InferenceModel):
    def sync_model(self, model_version, update_weight_args_list):
        return True

    def get_model_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        update_with_checkpoint: bool = True,
    ) -> None:
        pass

    def has_api_server(self) -> bool:
        return True

    def api_server_ready(self) -> Tuple[str, str]:
        return "http://localhosts:12345", "placeholder"


def generate_tasks(total_num: int, timeout_num: int = 0, exception_num: int = 0):
    tasks = [Task(workflow=DummyWorkflow, raw_task={}) for _ in range(total_num)]
    tasks.extend(
        [
            Task(
                workflow=DummyWorkflow,
                raw_task={"error_type": "timeout", "timeout": 5},
            )
            for _ in range(timeout_num)
        ]
    )
    tasks.extend(
        [
            Task(
                workflow=DummyWorkflow,
                raw_task={"error_type": "exception"},
            )
            for _ in range(exception_num)
        ]
    )
    return tasks


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.explorer.max_retry_times = 1
        self.config.explorer.max_timeout = 5
        self.config.explorer.runner_per_model = 2
        self.config.buffer.read_batch_size = 2
        self.config.buffer.pad_token_id = 0
        self.config.buffer.explorer_output = (
            self.config.buffer.trainer_input.experience_buffer
        ) = StorageConfig(
            name="test",
            storage_type=StorageType.QUEUE,
            algorithm_type="ppo",
            path="",
        )
        self.queue = QueueReader(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )

    async def test_scheduler(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(8)
        scheduler.schedule(tasks, step=0)
        self.assertTrue(scheduler.has_step(0))
        results = await scheduler.get_results(step=0, min_num=8, timeout=20)
        self.assertEqual(len(results), 8)
        scheduler.schedule(tasks, step=1)
        scheduler.schedule(tasks[:4], step=2)
        self.assertFalse(scheduler.has_step(0))
        results = await scheduler.get_results(step=0, min_num=8)
        self.assertFalse(scheduler.has_step(0))
        self.assertEqual(len(results), 0)  # step 0 has no more tasks
        self.assertFalse(scheduler.has_step(0))
        self.assertTrue(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        await scheduler.wait_all()
        st = time.time()
        results = await scheduler.get_results(step=1)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(results), 8)
        self.assertFalse(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        st = time.time()
        results = await scheduler.get_results(step=2)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(results), 4)
        self.assertFalse(scheduler.has_step(2))
        await scheduler.stop()
