import os
from typing import Any, Dict

from trinity.common.config import Config, OperatorConfig, TaskPipelineConfig
from trinity.utils.log import get_logger


def check_and_run_task_pipeline(config: Config):
    if not (config.mode == "explore" or config.mode == "bench"):
        # task pipeline is only available when using Explorer
        return
    if config.data_processor.task_pipeline is None:
        return
    for input in config.data_processor.task_pipeline.inputs.values():
        if not input.path:
            raise ValueError("`path` is required for each `data_processor.task_pipeline.inputs`.")
        if not os.path.exists(input.path):
            raise FileNotFoundError(f"{input.path} does not exist.")
        if not os.path.isfile(input.path):
            raise ValueError(
                f"{input.path} is not a file. Currently, task pipeline only support process on file."
            )

    try:
        task_pipeline = TaskPipeline(config)
        task_pipeline.process()
    except Exception as e:
        raise RuntimeError(f"Task pipeline failed: {e}")


class TaskPipeline:
    """
    A class to process task datasets through DataJuicer.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        from trinity.service.data_juicer.client import DataJuicerClient

        self.client = DataJuicerClient(config.service.data_juicer)  # type: ignore [arg-type]
        self.pipeline_config = config.data_processor.task_pipeline

    def convert_pipeline_config(self, pipeline_config: TaskPipelineConfig) -> Dict[str, Any]:
        """
        Convert the TaskPipelineConfig to a format suitable for DataJuicer.
        """

        def _convert_operator(operator: OperatorConfig) -> Dict:
            return {operator.name: {key: value for key, value in operator.args.items()}}

        converted_config = {
            "pipeline_type": "task",
            "operators": [_convert_operator(op) for op in pipeline_config.operators],
            "config_path": pipeline_config.config_path,
            "np": pipeline_config.np,
            "inputs": [input.path for input in pipeline_config.inputs.values()],
        }
        return converted_config

    def process(self) -> Dict[str, Any]:
        """
        Process the task datasets using DataJuicer.

        Returns:
            Dict[str, Any]: Metrics for logging.
        """
        # Convert the pipeline configuration
        converted_config = self.convert_pipeline_config(self.pipeline_config)  # type: ignore [arg-type]
        self.client.initialize(converted_config)
        self.client.process_task()
        return {}
