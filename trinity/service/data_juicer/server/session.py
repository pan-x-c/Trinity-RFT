from typing import Dict, Tuple

from datasets import Dataset

from .config_parser import ConfigParser
from .utils import DataJuicerConfigModel


class DataJuicerSession:
    """
    A session for interacting with the Data-Juicer service.
    This class manages the connection and provides methods to send and receive data.
    """

    def __init__(self, config: DataJuicerConfigModel, agent: ConfigParser):
        """
        Initialize the DataJuicerSession with a URL and configuration.

        Args:
            config (DataJuicerConfigModel): Configuration parameters provided by Trinity.
        """
        self.config = config
        self.agent = agent
        self.agent.parse(config)

    def process(self, ds: Dataset) -> Tuple[Dataset, Dict]:
        # TODO: Implement the processing logic using data juicer executor
        return ds, {}
