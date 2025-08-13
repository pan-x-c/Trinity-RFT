import time
import unittest
from multiprocessing import Process

try:
    import data_juicer  # noqa: [F401]
except ImportError:
    raise ImportError(
        "data_juicer module is not installed. Please install it with `pip install py-data-juicer` to run the tests."
    )
from jsonargparse import Namespace

from trinity.service.data_juicer.client import DataJuicerClient
from trinity.service.data_juicer.server.server import main
from trinity.service.data_juicer.server.utils import DataJuicerConfigModel, parse_config
from trinity.utils.distributed import get_available_port


class TestDataJuicer(unittest.TestCase):
    def test_config(self):
        trinity_config = {
            "operators": [
                {
                    "llm_quality_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "input_keys": ["prompt_text"],
                        "field_names": ["prompt", "response"],
                    }
                },
                {
                    "llm_difficulty_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "enable_vllm": False,
                    }
                },
            ]
        }
        config = DataJuicerConfigModel.model_validate(trinity_config)
        dj_config = parse_config(config)
        self.assertIsInstance(dj_config, Namespace)

    def test_data_juicer_experience(self):
        with self.assertRaises(ConnectionError):
            # server is not running, should raise an error
            DataJuicerClient(url="http://localhost:5005")

        # Start the server in a separate process
        def start_server(port):
            server_process = Process(
                target=main, kwargs={"host": "localhost", "port": port, "debug": False}
            )
            server_process.start()
            return server_process

        port = get_available_port()
        server_process = start_server(port)
        time.sleep(15)  # Wait for the server to start
        client = DataJuicerClient(url=f"http://localhost:{port}")
        client.initialize(
            {
                "operators": [
                    {
                        "llm_quality_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "input_keys": ["prompt_text"],
                            "field_names": ["prompt", "response"],
                        }
                    },
                    {
                        "llm_difficulty_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "enable_vllm": False,
                        }
                    },
                ]
            }
        )
        server_process.terminate()
        server_process.join()
