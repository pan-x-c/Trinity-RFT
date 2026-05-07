from __future__ import annotations

import asyncio
import os
from logging import Logger
from typing import List, Literal, Optional, Sequence, Tuple

import httpx
import torch
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.model import BaseInferenceModel
from trinity.common.models.sglang_patch import get_api_server
from trinity.manager.synchronizer import Synchronizer


class SGLangClient:
    """A simple http client to interact with the SGLang API server."""

    def __init__(self, server_url: str, api_key: Optional[str], logger: Logger):
        self.server_url = server_url
        self.api_key = api_key
        self.logger = logger

    async def _server_call(
        self,
        method: Literal["GET", "POST"],
        endpoint: str,
        payload: Optional[dict] = None,
        timeout: float = 60,
    ) -> dict:
        async with httpx.AsyncClient(
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            }
        ) as client:
            url = f"{self.server_url}{endpoint}"
            self.logger.debug(
                f"Making {method} request to SGLang API server at {url} with payload: {payload}"
            )
            try:
                if method == "GET":
                    response = await client.get(url, timeout=timeout)
                elif method == "POST":
                    response = await client.post(url, json=payload or {}, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.error(
                    f"Error during {method} request to SGLang API server at {url}: {e}"
                )
                return {"error": str(e)}

    async def health_check(self) -> bool:
        response = await self._server_call("GET", "/health")
        return response.get("status") == "ok"

    async def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
    ) -> bool:
        payload = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        }
        response = await self._server_call(
            "POST", "/init_weights_update_group", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to initialize weights update group in SGLang API server: {response.get('message')}"
            )
        return success

    async def destroy_weights_update_group(self, group_name: str) -> bool:
        payload = {"group_name": group_name}
        response = await self._server_call("POST", "/destroy_weights_update_group", payload)
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to destroy weights update group in SGLang API server: {response.get('message')}"
            )
        return success

    async def update_weights_from_distributed(
        self,
        state_dict_meta_list: List[Tuple[str, str, Tuple]],
        group_name: str,
        flash_cache: bool = True,
        abort_all_requests: bool = True,
        weight_version: Optional[str] = None,
        timeout: float = 300,
    ) -> bool:
        payload = {
            "state_dict_meta_list": state_dict_meta_list,
            "group_name": group_name,
            "flash_cache": flash_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
        }
        response = await self._server_call(
            "POST", "/update_weights_from_distributed", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to update weights from distributed in SGLang API server: {response.get('message')}"
            )
        return success

    async def update_weights_from_disk(
        self,
        model_path: str,
        abort_all_requests: bool = True,
        weight_version: Optional[str] = None,
        is_async: bool = False,
        timeout: float = 300,
    ) -> bool:
        payload = {
            "model_path": model_path,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
            "is_async": is_async,
            "torch_empty_cache": True,
        }
        response = await self._server_call(
            "POST", "/update_weights_from_disk", payload, timeout=timeout
        )
        success = response.get("success", False)
        if not success:
            self.logger.error(
                f"Failed to update weights from disk in SGLang API server: {response.get('message')}"
            )
        return success


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
        self.api_server: Optional[asyncio.Task[None]] = None
        self.api_client: Optional[SGLangClient] = None
        self.synchronizer = None
        self.model_version = 0
        self.server_args: Optional[ServerArgs] = None
        self._prepared = False
        self.async_lock = asyncio.Lock()

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explorer_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: List = None,
    ):
        assert (
            self.api_client is not None
        ), "API client must be initialized before calling init_process_group"
        if not self.synchronizer:
            self.synchronizer = Synchronizer.get_actor(namespace=self.config.ray_namespace)
        self.state_dict_meta = state_dict_meta
        return await self.api_client.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
            timeout=timeout,
        )

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
            "skip_server_warmup": False,
            "disable_piecewise_cuda_graph": True,
            "api_key": "EMPTY",
            "device": "cuda",
            "mamba_scheduler_strategy": "extra_buffer",
        }
        # if self.config.chat_template:
        #     server_args_kwargs["chat_template"] = self.config.chat_template
        # TODO: fill in nnodes and node_rank for distributed setups
        return ServerArgs(**server_args_kwargs)

    def _get_api_server_exit_reason(self) -> Optional[str]:
        if self.api_server is None or not self.api_server.done():
            return None
        if self.api_server.cancelled():
            return "cancelled"
        exc = self.api_server.exception()
        return "unknown error" if exc is None else repr(exc)

    async def _wait_until_server_ready(self, server_url: str) -> None:
        max_retries = 100
        assert self.server_args is not None and self.api_client is not None
        for _ in range(max_retries):
            reason = self._get_api_server_exit_reason()
            if reason is not None:
                raise RuntimeError(f"SGLang API server exited before becoming ready: {reason}.")
            if await self.api_client.health_check():
                self.logger.info(f"SGLang API server at {server_url} is ready.")
                return
            self.logger.debug(f"SGLang API server at {server_url} not ready yet, retrying...")
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
        self.api_server = get_api_server(self.server_args, logger=self.logger)
        self.api_client = SGLangClient(
            server_url=f"http://{self.api_server_host}:{self.api_server_port}",
            api_key=self.server_args.api_key,
            logger=self.logger,
        )
        await self._wait_until_server_ready(self.server_args.url())
        return True

    async def shutdown(self) -> None:
        if self.api_server is not None:
            self.api_server.cancel()
            try:
                await self.api_server
            except asyncio.CancelledError:
                pass
            reason = self._get_api_server_exit_reason()
            if reason not in {None, "cancelled"}:
                self.logger.warning("Embedded SGLang HTTP server exited with error: %s", reason)
            self.api_server = None

    async def sync_model(
        self, model_version: int, method: SyncMethod, timeout: float = 1200
    ) -> int:
        assert (
            self.api_client is not None
        ), "API client must be initialized before calling sync_model"
        assert (
            self.synchronizer is not None
        ), "Synchronizer must be initialized before calling sync_model"
        if method == SyncMethod.NCCL:
            await self.api_client.update_weights_from_distributed(
                state_dict_meta_list=self.state_dict_meta,
                group_name="weights_update",
                weight_version=str(model_version),
                timeout=timeout,
            )
        elif method == SyncMethod.CHECKPOINT:
            model_path = await self.synchronizer.get_latest_model_path.remote()
            if model_path is not None:
                await self.api_client.update_weights_from_disk(
                    model_path=model_path,
                    weight_version=str(model_version),
                    timeout=timeout,
                )
        else:
            raise ValueError(f"Unsupported sync method for SGLang: {method}")
        return model_version

    def get_model_version(self) -> int:
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        if not self._prepared:
            raise RuntimeError("Model is not prepared. Please call `prepare()` first.")
        return f"http://{self.api_server_host}:{self.api_server_port}"
