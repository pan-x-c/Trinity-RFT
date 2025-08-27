"""Ray Queue storage"""
import asyncio
import time
from collections import deque
from copy import deepcopy
from typing import List

import ray

from trinity.buffer.queue import QueueBuffer
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


class QueueStorage:
    """An wrapper of a async queue."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(f"queue_{storage_config.name}")
        self.config = config
        self.capacity = storage_config.capacity
        self.queue = QueueBuffer.get_queue(storage_config, config)
        st_config = deepcopy(storage_config)
        st_config.wrap_in_ray = False
        if st_config.path is not None:
            if is_database_url(st_config.path):
                from trinity.buffer.writer.sql_writer import SQLWriter

                st_config.storage_type = StorageType.SQL
                self.writer = SQLWriter(st_config, self.config)
            elif is_json_file(st_config.path):
                from trinity.buffer.writer.file_writer import JSONWriter

                st_config.storage_type = StorageType.FILE
                self.writer = JSONWriter(st_config, self.config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            from trinity.buffer.writer.file_writer import JSONWriter

            st_config.storage_type = StorageType.FILE
            self.writer = JSONWriter(st_config, self.config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0
        self.exp_pool = deque()  # A pool to store experiences
        self.closed = False

    async def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            await self.queue.close()
            await self.writer.release()
        return self.ref_count

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, exp_list: List) -> None:
        """Put batch of experience."""
        await self.queue.put(exp_list)
        if self.writer is not None:
            self.writer.write(exp_list)

    async def get_batch(self, batch_size: int, timeout: float) -> List:
        """Get batch of experience."""
        start_time = time.time()
        while len(self.exp_pool) < batch_size:
            if self.queue.stopped():
                # If the queue is stopped, ignore the rest of the experiences in the pool
                raise StopAsyncIteration("Queue is closed and no more items to get.")
            try:
                exp_list = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.exp_pool.extend(exp_list)
            except asyncio.TimeoutError:
                if time.time() - start_time > timeout:
                    self.logger.error(
                        f"Timeout when waiting for experience, only get {len(self.exp_pool)} experiences.\n"
                        "This phenomenon is usually caused by the workflow not returning enough "
                        "experiences or running timeout. Please check your workflow implementation."
                    )
                    batch = list(self.exp_pool)
                    self.exp_pool.clear()
                    return batch
        return [self.exp_pool.popleft() for _ in range(batch_size)]

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        """Get the queue actor."""
        return (
            ray.remote(cls)
            .options(
                name=f"queue-{storage_config.name}",
                namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
            )
            .remote(storage_config, config)
        )
