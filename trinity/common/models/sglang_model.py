from __future__ import annotations

import asyncio
import multiprocessing
import os
from typing import List, Optional, Sequence

import httpx
import torch
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import BaseInferenceModel


class SGLangRolloutModel(BaseInferenceModel):
    """Wrapper around the SGLang engine to handle async requests.

    Args:
        config (Config): The config.
    """

    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        super().__init__(config)
        if config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        if not self.config.enable_openai_api:
            self.logger.warning("SGLangRolloutModel requires OpenAI API to be enabled.")
            self.config.enable_openai_api = True
        self.api_server_host: Optional[str] = None
        self.api_server_port: Optional[int] = None
        self.server_process: Optional[multiprocessing.Process] = None
        self.model_version = 0
        self.server_args: Optional[ServerArgs] = None
        self._prepared = False
        self.async_lock = asyncio.Lock()

    async def _initialize_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer.truncation_side = "left"

    async def prepare(self) -> None:
        async with self.async_lock:
            if self._prepared:
                return
            await self.run_api_server()
            self._prepared = True

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        raise NotImplementedError(
            "SGLangRolloutModel currently only supports OpenAI API access via ModelWrapper clients."
        )

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        raise NotImplementedError(
            "SGLangRolloutModel currently only supports OpenAI API access via ModelWrapper clients."
        )

    async def logprobs(self, token_ids: List[int], **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "SGLangRolloutModel does not implement local logprobs in the OpenAI-API-only integration."
        )

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools=None,
        temperature: Optional[float] = None,
    ) -> Experience:
        del messages, tools, temperature
        raise NotImplementedError(
            "SGLangRolloutModel does not implement local experience conversion in the OpenAI-API-only integration."
        )

    def _build_server_args(self, host: str, port: int) -> ServerArgs:
        server_args_kwargs = {
            "model_path": self.config.model_path,
            "host": host,
            "port": port,
            "tp_size": self.config.tensor_parallel_size,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "served_model_name": self.config.name or self.config.model_path,
            "trust_remote_code": self.config.trust_remote_code,
            "context_length": self.config.max_model_len,
            "enable_multimodal": self.config.enable_multimodal,
            "skip_server_warmup": True,
            "disable_piecewise_cuda_graph": True,
            "api_key": "EMPTY",
            "device": "cuda",
        }
        # if self.config.chat_template:
        #     server_args_kwargs["chat_template"] = self.config.chat_template
        # TODO: fill in nnodes and node_rank for distributed setups
        return ServerArgs(**server_args_kwargs)

    async def _wait_until_server_ready(self, server_url: str) -> None:
        max_retries = 100
        assert self.server_args is not None
        async with httpx.AsyncClient(
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.server_args.api_key}",
            }
        ) as client:
            for _ in range(max_retries):
                if self.server_process is not None and not self.server_process.is_alive():
                    raise RuntimeError("SGLang API server exited before becoming ready.")
                try:
                    response = await client.get(f"{server_url}/health", timeout=5)
                    if response.status_code == 200:
                        return
                except Exception:
                    self.logger.debug("SGLang API server not ready yet, retrying...")
                await asyncio.sleep(2)
        self.logger.error(
            f"SGLang API server at {server_url} not ready after {max_retries} attempts."
        )
        await self.shutdown()
        raise RuntimeError(
            f"SGLang API server at {server_url} not ready after {max_retries} attempts."
        )

    async def run_api_server(self) -> bool:
        if self.api_server_host is None or self.api_server_port is None:
            self.api_server_host, self.api_server_port = self.get_available_address()
        self.server_args = self._build_server_args(
            host=self.api_server_host,
            port=self.api_server_port,
        )
        self.server_process = multiprocessing.Process(
            target=launch_server, args=(self.server_args,)
        )
        self.server_process.start()
        await self._wait_until_server_ready(self.server_args.url())
        return True

    async def shutdown(self) -> None:
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.join()
            self.server_process = None

    async def sync_model(self, model_version: int) -> int:
        raise NotImplementedError("TO BE DONE")

    def get_model_version(self) -> int:
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        if not self._prepared:
            raise RuntimeError("Model is not prepared. Please call `prepare()` first.")
        return f"http://{self.api_server_host}:{self.api_server_port}"
