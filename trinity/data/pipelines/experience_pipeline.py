from typing import Dict, List

import ray

from trinity.buffer.buffer import get_buffer_writer
from trinity.common.config import BufferConfig, ExperiencePipelineConfig
from trinity.common.experience import Experience
from trinity.data.operators.experience_operator import ExperienceOperator
from trinity.utils.log import get_logger


def get_input_buffers(
    pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig
) -> Dict:
    """Get input buffers for the experience pipeline."""
    input_buffers = {}
    for input_name, input_config in pipeline_config.inputs.items():
        buffer_writer = get_buffer_writer(input_config, buffer_config)
        input_buffers[input_name] = buffer_writer
    return input_buffers


class ExperiencePipeline:
    """
    A class to process experiences in a distributed manner using Ray.
    """

    def __init__(self, pipeline_config: ExperiencePipelineConfig, buffer_config: BufferConfig):
        self.logger = get_logger(__name__)
        self.inputs = get_input_buffers(pipeline_config, buffer_config)
        self.operators = ExperienceOperator.create_operators(pipeline_config.operators)
        self.output = get_buffer_writer(
            pipeline_config.output,  # type: ignore [arg-type]
            buffer_config,
        )

    @classmethod
    def get_ray_actor(cls, pipeline_config: ExperiencePipelineConfig):
        """Get a Ray actor for the experience pipeline."""
        return (
            ray.remote(cls)
            .options(name="ExperiencePipeline", namespace=pipeline_config.ray_namespace)
            .remote(pipeline_config)
        )

    def read_from_input_buffers(self) -> List[Experience]:
        # TODO
        return []

    def run(self) -> None:
        """Run the experience pipeline."""
        while True:
            # Read experiences from input buffers
            try:
                exps = self.read_from_input_buffers()
            except StopIteration:
                self.logger.info("No more experiences to read from input buffers.")
                break

            # Process experiences through operators
            for operator in self.operators:
                exps = operator.process(exps)

            # Write processed experiences to output buffer
            self.output.write(exps)
