from abc import ABC, abstractmethod
from typing import List

from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experience
from trinity.utils.registry import Registry

SAMPLE_STRATEGY = Registry("sample_strategy")


class SampleStrategy(ABC):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        self.buffer_config = buffer_config

    @abstractmethod
    def sample(self, step: int, **kwargs) -> List[Experience]:
        """Sample experiences from buffer.

        Args:
            step (`int`): The step number of current step.
        """

    @classmethod
    def default_args(cls) -> dict:
        return {}


@SAMPLE_STRATEGY.register_module("warmup")
class WarmupSampleStrategy(SampleStrategy):
    """The default sample strategy."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.experience_buffer, buffer_config  # type: ignore
        )
        self.sft_warmup_steps = buffer_config.trainer_input.sft_warmup_steps
        if self.sft_warmup_steps > 0 and buffer_config.trainer_input.sft_warmup_dataset is None:
            raise ValueError("sft_warmup_dataset is required when sft_warmup_steps > 0")
        if buffer_config.trainer_input.sft_warmup_dataset is not None:
            self.sft_buffer = get_buffer_reader(
                buffer_config.trainer_input.sft_warmup_dataset, buffer_config
            )
        else:
            self.sft_buffer = None

    def sample(self, step: int, **kwargs) -> List[Experience]:
        if step <= self.sft_warmup_steps:
            return self.sft_buffer.read()
        else:
            return self.exp_buffer.read()


@SAMPLE_STRATEGY.register_module("default")
class DefaultSampleStrategy(SampleStrategy):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.experience_buffer, buffer_config  # type: ignore
        )

    def sample(self, step: int, **kwargs) -> List[Experience]:
        return self.exp_buffer.read()
