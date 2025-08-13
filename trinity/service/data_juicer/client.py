import io
import time
from multiprocessing import Process
from typing import Dict, List, Tuple

import requests
from datasets import Dataset

from trinity.common.config import DataJuicerServiceConfig
from trinity.common.experience import Experience, from_hf_datasets, to_hf_datasets
from trinity.utils.distributed import get_available_port


class DataJuicerClient:
    """Client for interacting with the DataJuicer server."""

    def __init__(self, config: DataJuicerServiceConfig):
        self.config = config
        self.url = config.server_url
        self.session_id = None
        self.server = None
        if not self.config.auto_start:
            # If auto-start is disabled, check the connection immediately
            self._check_connection()

    def _start_server(self):
        """Start the DataJuicer server."""
        if not self.config.auto_start:
            # Server auto-start is disabled, use the provided URL
            return None

        from trinity.service.data_juicer.server.server import main

        if not self.config.port:
            self.config.port = get_available_port()
        self.url = f"http://localhost:{self.config.port}"
        server_process = Process(
            target=main, kwargs={"host": "localhost", "port": self.config.port, "debug": False}
        )
        server_process.start()
        # Wait for the server to start
        while True:
            try:
                if self._check_connection():
                    break
            except ConnectionError:
                time.sleep(5)
        return server_process

    def _check_connection(self) -> bool:
        """Check if the DataJuicer server is reachable."""
        try:
            response = requests.get(f"{self.url}/health")  # Check if the server is running
        except Exception as e:
            raise ConnectionError(f"Failed to connect to DataJuicer server at {self.url}: {e}")
        if response.status_code != 200:
            raise ConnectionError(
                f"DataJuicer server at {self.url} is not reachable. Status code: {response.status_code}"
            )
        return True

    def initialize(self, config: dict):
        self.server = self._start_server()
        response = requests.post(f"{self.url}/create", json=config)
        response.raise_for_status()
        self.session_id = response.json().get("session_id")

    def process_experience(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if not self.session_id:
            raise ValueError("DataJuicer session is not initialized.")

        hf_exps = to_hf_datasets(exps)

        # Serialize to an in-memory buffer
        buffer = io.BytesIO()
        hf_exps.to_parquet(buffer)
        buffer.seek(0)

        # The filename in the multipart-form data can be a constant string
        files = {"file": ("experiences.parquet", buffer, "application/octet-stream")}
        data = {"session_id": self.session_id}
        response = requests.post(f"{self.url}/process", data=data, files=files)
        response.raise_for_status()

        # Deserialize from the response content in-memory
        metrics = response.json().get("metrics", {})
        with io.BytesIO(response.content) as recv_buffer:
            ds = Dataset.from_parquet(recv_buffer)
            return from_hf_datasets(ds), metrics

    def close(self):
        """Close the DataJuicer client connection."""
        if self.session_id:
            response = requests.post(f"{self.url}/close", json={"session_id": self.session_id})
            response.raise_for_status()
            self.session_id = None
        if self.server:
            self.server.terminate()
            self.server.join()
            self.server = None
