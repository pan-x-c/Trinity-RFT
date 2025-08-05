from typing import Dict, List

import ray

from trinity.buffer.buffer import get_buffer_reader, get_buffer_writer
from trinity.buffer.ray_wrapper import is_database_url, is_json_file
from trinity.common.config import BufferConfig, ExperiencePipelineConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.data.operators.experience_operator import ExperienceOperator
from trinity.utils.log import get_logger


def get_input_buffers(
    pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig
) -> Dict:
    """Get input buffers for the experience pipeline."""
    input_buffers = {}
    for input_name, input_config in pipeline_config.inputs.items():
        buffer_reader = get_buffer_reader(input_config, buffer_config)
        input_buffers[input_name] = buffer_reader
    return input_buffers


class ExperiencePipeline:
    """
    A class to process experiences.
    """

    def __init__(self, pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig):
        self.logger = get_logger(__name__)
        self.operators = ExperienceOperator.create_operators(pipeline_config.operators)
        self.input_store = None
        if pipeline_config.save_input:
            if is_json_file(pipeline_config.input_save_path):  # type: ignore [arg-type]
                self.input_store = get_buffer_writer(
                    StorageConfig(
                        storage_type=StorageType.FILE,
                        path=pipeline_config.input_save_path,
                    ),
                    buffer_config,
                )
            elif is_database_url(pipeline_config.input_save_path):  # type: ignore [arg-type]
                self.input_store = get_buffer_writer(
                    StorageConfig(
                        storage_type=StorageType.SQL,
                        path=pipeline_config.input_save_path,
                    ),
                    buffer_config,
                )
            else:
                raise ValueError(
                    f"Unsupported save_input format: {pipeline_config.save_input}. "
                    "Only JSON file path or SQLite URL is supported."
                )
        self.output = get_buffer_writer(
            pipeline_config.output,  # type: ignore [arg-type]
            buffer_config,
        )

    async def prepare(self) -> None:
        await self.output.acquire()

    @classmethod
    def get_ray_actor(cls, pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig):
        """Get a Ray actor for the experience pipeline."""
        return (
            ray.remote(cls)
            .options(name="ExperiencePipeline", namespace=pipeline_config.ray_namespace)
            .remote(pipeline_config, buffer_config)
        )

    async def run(self, exps: List[Experience]) -> None:
        """Run the experience pipeline."""
        if self.input_store is not None:
            await self.input_store.write_async(exps)

        # Process experiences through operators
        for operator in self.operators:
            exps = operator.process(exps)

        # Write processed experiences to output buffer
        self.output.write(exps)

    async def close(self) -> None:
        await self.output.release()
