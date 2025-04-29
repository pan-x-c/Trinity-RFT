# Tests for explorer
import unittest
from typing import List

import ray
import torch
from transformers import AutoTokenizer

from trinity.cli.launcher import explore
from trinity.common.config import Config
from trinity.common.constants import MonitorType
from trinity.common.experience import Experience, Experiences
from trinity.common.models.model import InferenceModel, ModelWrapper
from trinity.utils.log import get_logger

from ..tools import (
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)


@ray.remote
class DummyModel(InferenceModel):
    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.logger.info("DummyModel init")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
        self.repeat_times = config.explorer.repeat_times

    def sync_model(self, update_weight_args_list):
        self.logger.info("DummyModel sync model weight")
        return True

    def get_ckp_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        offline_update: bool = True,
    ) -> None:
        self.logger.info("DummyModel init process group")

    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        self.logger.info("DummyModel chat async")
        dummy_reply = "This is a dummy reply"
        tokens = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": dummy_reply}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return [
            Experience(
                tokens=tokens[0],
                prompt_length=len(prompt_tokens["input_ids"][0]),
                logprobs=torch.zeros_like(tokens[0]),
                response_text=dummy_reply,
                reward=0.1,
            )
            for _ in range(self.repeat_times)
        ]


def create_dummy_rollout_models(config: Config) -> List[InferenceModel]:
    return [DummyModel.remote(config) for _ in range(config.explorer.engine_num)]


class DummyModelTest(unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm_async"
        self.config.explorer.repeat_times = 3

    def test_dummy_model(self):
        """Test the dummy model."""
        model = DummyModel.remote(self.config)
        model = ModelWrapper(model, model_type="vllm_async")
        responses = model.chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(len(responses), self.config.explorer.repeat_times)
        for resp in responses:
            self.assertEqual(resp.response_text, "This is a dummy reply")
            self.assertEqual(len(resp.tokens), len(resp.logprobs))
        res = Experiences.gather_experiences(responses)
        self.assertEqual(res.batch_size, self.config.explorer.repeat_times)


class BaseExplorerCase:
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm_async"
        self.config.explorer.repeat_times = 4
        self.config.monitor.monitor_type = MonitorType.TENSORBOARD
        self.config.monitor.project = "Trinity-unittest"
        self.config.monitor.name = "explorer-unittest"
        self.config.model.checkpoint_path = get_checkpoint_path()
        self.config.data = get_unittest_dataset_config("countdown")
        self.config.check_and_update()


class TestExplorerCountdown(BaseExplorerCase, unittest.TestCase):
    def test_explorer(self):
        self.config.data = get_unittest_dataset_config("countdown")
        explore(self.config)
