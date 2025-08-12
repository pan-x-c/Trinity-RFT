from typing import Dict, List, Optional, Tuple

from trinity.buffer.operators.experience_operator import (
    EXPERIENCE_OPERATORS,
    ExperienceOperator,
)
from trinity.common.experience import Experience
from trinity.service.data_juicer.client import DataJuicerClient


@EXPERIENCE_OPERATORS.register_module("data_juicer")
class DataJuicerOperator(ExperienceOperator):
    def __init__(
        self,
        data_juicer_url: str,
        operators: Optional[List[Dict]] = None,
        config_path: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize the DataJuicerOperator with a URL and configuration.

        Args:
            data_juicer_url (str): The URL of the Data-Juicer server.
            operators(`List[Dict]`): A list of operators with their configurations.
            config_path(`str`): Path to the Data-Juicer configuration file.
            description(`str`): The operator you want to use, described in natural language (Experimental).

        Note:
            - Must include one of the following, and the priority is from high to low:
                - `operators` (`List[Dict]`)
                - `config_path` (`str`)
                - `description` (`str`)
        """
        self.client = DataJuicerClient(
            url=data_juicer_url,
        )
        self.client.initialize(
            {"operators": operators, "config_path": config_path, "description": description}
        )

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        return self.client.process(exps)

    def close(self):
        """Close the DataJuicer client connection."""
        self.client.close()
