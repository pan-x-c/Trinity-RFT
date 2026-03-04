"""Writer of the Queue buffer."""
import asyncio
from typing import Any, Dict, List, cast

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.queue import QueueStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience


class QueueWriter(BufferWriter):
    """Writer of the Queue buffer."""

    def __init__(self, config: StorageConfig):
        assert config.storage_type == StorageType.QUEUE.value
        self.queue = QueueStorage.get_wrapper(config)
        self._use_zmq = False
        self._zmq_socket = None

        zmq_config = config.zmq
        if zmq_config is not None and zmq_config.enable:
            endpoints = cast(Dict[str, Any], ray.get(self.queue.get_zmq_endpoints.remote()))
            if endpoints.get("enabled", False):
                import zmq

                self._zmq_socket = zmq.Context.instance().socket(zmq.PUSH)
                self._zmq_socket.setsockopt(zmq.SNDHWM, zmq_config.sndhwm)
                self._zmq_socket.setsockopt(zmq.LINGER, zmq_config.linger_ms)
                self._zmq_socket.connect(endpoints["writer_endpoint"])
                self._use_zmq = True

    def write(self, data: List) -> None:
        if self._use_zmq:
            assert self._zmq_socket is not None
            payload = Experience.serialize_many(data)
            self._zmq_socket.send(payload)
            return
        ray.get(self.queue.put_batch.remote(data))

    async def write_async(self, data):
        if self._use_zmq:
            return await asyncio.to_thread(self.write, data)
        return await self.queue.put_batch.remote(data)

    async def acquire(self) -> int:
        return await self.queue.acquire.remote()

    async def release(self) -> int:
        if self._zmq_socket is not None:
            self._zmq_socket.close(0)
            self._zmq_socket = None
        return await self.queue.release.remote()

    def __del__(self):
        if self._zmq_socket is not None:
            self._zmq_socket.close(0)
            self._zmq_socket = None
