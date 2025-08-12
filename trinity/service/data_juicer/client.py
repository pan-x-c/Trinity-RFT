from typing import List

import requests

from trinity.common.experience import Experience, from_hf_datasets, to_hf_datasets


class DataJuicerClient:
    def __init__(self, url: str = "http://localhost:5005"):
        self.url = url
        self.session_id = None

    def initialize(self, config: dict):
        response = requests.post(f"{self.url}/create", json=config)
        response.raise_for_status()
        self.session_id = response.json().get("session_id")

    def process(self, exps: List[Experience]) -> List[Experience]:
        import os
        import tempfile

        from datasets import Dataset

        if not self.session_id:
            raise ValueError("DataJuicer session is not initialized.")
        hf_exps = to_hf_datasets(exps)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            hf_exps.to_parquet(tmp_file.name)
            tmp_file_path = tmp_file.name
        try:
            with open(tmp_file_path, "rb") as f:
                files = {"file": (os.path.basename(tmp_file_path), f, "application/octet-stream")}
                data = {"session_id": self.session_id}
                response = requests.post(f"{self.url}/process", data=data, files=files)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as recv_file:
                recv_file.write(response.content)
                recv_file_path = recv_file.name
            try:
                ds = Dataset.from_parquet(recv_file_path)
                return from_hf_datasets(ds)
            finally:
                os.remove(recv_file_path)
        finally:
            os.remove(tmp_file_path)

    def close(self):
        """Close the DataJuicer client connection."""
        if self.session_id:
            requests.post(f"{self.url}/close", json={"session_id": self.session_id})
