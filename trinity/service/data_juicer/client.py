import io
from typing import Dict, List, Tuple

import requests
from datasets import Dataset

from trinity.common.experience import Experience, from_hf_datasets, to_hf_datasets


class DataJuicerClient:
    def __init__(self, url: str = "http://localhost:5005"):
        self.url = url
        self.session_id = None
        try:
            response = requests.get(f"{self.url}/health")  # Check if the server is running
        except Exception:
            raise ConnectionError(f"Failed to connect to DataJuicer server at {self.url}")
        response.raise_for_status()

    def initialize(self, config: dict):
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
