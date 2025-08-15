from typing import Any, Dict, List, Optional

from trinity.common.config import Config, TaskPipelineConfig, OperatorConfig
from trinity.utils.log import get_logger


class TaskPipeline:
    """
    A class to process task datasets through DataJuicer.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        from trinity.service.data_juicer.client import DataJuicerClient

        self.client = DataJuicerClient(config.service.data_juicer)
        self.pipeline_config = config.data_processor.task_pipeline


    def convert_pipeline_config(
        self, pipeline_config: TaskPipelineConfig
    ) -> Dict[str, Any]:
        """
        Convert the TaskPipelineConfig to a format suitable for DataJuicer.
        """
        def _convert_operator(operator: OperatorConfig) -> Dict:
            return {
                operator.name: {
                    key: value
                    for key, value in operator.args.items()
                }
            }
        converted_config = {
            "pipeline_type": "task",
            "operators": [_convert_operator(op) for op in pipeline_config.operators],
            "config_path": pipeline_config.config_path,
            "np": pipeline_config.np,
        }
        return converted_config

    def process(self) -> Dict[str, Any]:
        """
        Process the task datasets using DataJuicer.

        Returns:
            Dict[str, Any]: Metrics for logging.
        """
        # Convert the pipeline configuration
        converted_config = self.convert_pipeline_config(self.pipeline_config)
        self.client.initialize(converted_config)
        self.client.process_task()