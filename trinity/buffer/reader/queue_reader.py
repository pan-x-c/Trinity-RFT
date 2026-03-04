"""Reader of the Queue buffer."""

import asyncio

from typing import Dict, List, Optional
from typing import Any, cast

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.queue import QueueStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience


class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, config: StorageConfig):
        assert config.storage_type == StorageType.QUEUE.value
        self.timeout = config.max_read_timeout
        self.read_batch_size = config.batch_size
        self.queue = QueueStorage.get_wrapper(config)
        self._use_zmq = False
        self._zmq_socket = None

        zmq_config = config.zmq
        if zmq_config is not None and zmq_config.enable:
            endpoints = cast(Dict[str, Any], ray.get(self.queue.get_zmq_endpoints.remote()))
            if endpoints.get("enabled", False):
                import zmq.asyncio

                self._zmq_socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
                self._zmq_socket.setsockopt(zmq.SNDHWM, zmq_config.sndhwm)
                self._zmq_socket.setsockopt(zmq.RCVHWM, zmq_config.rcvhwm)
                self._zmq_socket.setsockopt(zmq.LINGER, zmq_config.linger_ms)
                self._zmq_socket.setsockopt(zmq.SNDTIMEO, int(config.max_read_timeout * 1000))
                self._zmq_socket.setsockopt(zmq.RCVTIMEO, int(config.max_read_timeout * 1000))
                self._zmq_socket.connect(endpoints["reader_endpoint"])
                self._use_zmq = True

    async def _read_via_zmq(self, batch_size: int, **kwargs) -> List:
        assert self._zmq_socket is not None
        min_model_version = int(kwargs.get("min_model_version", 0))
        request = {
            "cmd": "get_batch",
            "batch_size": batch_size,
            "timeout": float(self.timeout),
            "min_model_version": min_model_version,
        }
        await self._zmq_socket.send_json(request)
        status, payload = await self._zmq_socket.recv_multipart()
        status_text = status.decode("utf-8")

        if status_text == "ok":
            exps = Experience.deserialize_many(payload)
            if len(exps) != batch_size:
                raise TimeoutError(
                    f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
                )
            return exps

        if status_text == "eos":
            raise StopIteration()

        if status_text == "error":
            raise RuntimeError(payload.decode("utf-8"))

        raise RuntimeError(f"Unknown queue reader response status: {status_text}")

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        raise NotImplementedError("This function is deprecated, please use read_async instead.")

    async def read_async(self, batch_size: Optional[int] = None, **kwargs) -> List:
        batch_size = self.read_batch_size if batch_size is None else batch_size
        if self._use_zmq:
            try:
                return await self._read_via_zmq(batch_size, **kwargs)
            except StopIteration as e:
                raise StopAsyncIteration() from e

        exps = await self.queue.get_batch.remote(batch_size, timeout=self.timeout, **kwargs)
        if len(exps) != batch_size:
            raise TimeoutError(
                f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
            )
        return exps

    def state_dict(self) -> Dict:
        # Queue Not supporting state dict yet
        return {"current_index": 0}

    def load_state_dict(self, state_dict):
        # Queue Not supporting state dict yet
        return None

    def __del__(self):
        if self._zmq_socket is not None:
            self._zmq_socket.close(0)
            self._zmq_socket = None
