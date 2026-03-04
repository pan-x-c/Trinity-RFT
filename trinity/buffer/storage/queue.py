"""Ray Queue storage"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from sortedcontainers import SortedDict

from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


class PriorityFunction(ABC):
    """
    Each priority_fn,
        Args:
            item: List[Experience], assume that all experiences in it have the same model_version and use_count
            priority_fn_args: Dict, the arguments for priority_fn

        Returns:
            priority: float
            put_into_queue: bool, decide whether to put item into queue

    Note that put_into_queue takes effect both for new item from the explorer and for item sampled from the buffer.
    """

    @abstractmethod
    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        """Calculate the priority of item."""

    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict:
        """Return the default config."""


class LinearDecayPriority(PriorityFunction):
    """Calculate priority by linear decay.

    Priority is calculated as `model_version - decay * use_count. The item is always put back into the queue for reuse (as long as `reuse_cooldown_time` is not None).
    """

    def __init__(self, decay: float = 2.0):
        self.decay = decay

    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        priority = float(item[0].info["model_version"] - self.decay * item[0].info["use_count"])
        put_into_queue = True
        return priority, put_into_queue

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "decay": 2.0,
        }


class LinearDecayUseCountControlPriority(PriorityFunction):
    """Calculate priority by linear decay, use count control, and randomization.

    Priority is calculated as `model_version - decay * use_count`; if `sigma` is non-zero, priority is further perturbed by random Gaussian noise with standard deviation `sigma`.  The item will be put back into the queue only if use count does not exceed `use_count_limit`.
    """

    def __init__(self, decay: float = 2.0, use_count_limit: int = 3, sigma: float = 0.0):
        self.decay = decay
        self.use_count_limit = use_count_limit
        self.sigma = sigma

    def __call__(self, item: List[Experience]) -> Tuple[float, bool]:
        priority = float(item[0].info["model_version"] - self.decay * item[0].info["use_count"])
        if self.sigma > 0.0:
            priority += float(np.random.randn() * self.sigma)
        put_into_queue = (
            item[0].info["use_count"] < self.use_count_limit if self.use_count_limit > 0 else True
        )
        return priority, put_into_queue

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "decay": 2.0,
            "use_count_limit": 3,
            "sigma": 0.0,
        }


class QueueBuffer(ABC):
    async def set_min_model_version(self, min_model_version: int):
        self.min_model_version = max(min_model_version, 0)

    @abstractmethod
    async def put(self, exps: List[Experience]) -> None:
        """Put a list of experiences into the queue."""

    @abstractmethod
    async def get(self) -> List[Experience]:
        """Get a list of experience from the queue."""

    @abstractmethod
    def qsize(self) -> int:
        """Get the current size of the queue."""

    @abstractmethod
    async def close(self) -> None:
        """Close the queue."""

    @abstractmethod
    def stopped(self) -> bool:
        """Check if there is no more data to read."""

    @classmethod
    def get_queue(cls, config: StorageConfig) -> "QueueBuffer":
        """Get a queue instance based on the storage configuration."""
        logger = get_logger(__name__)
        replay_buffer = config.replay_buffer
        if replay_buffer is not None and replay_buffer.enable:
            capacity = config.capacity
            logger.info(
                f"Using AsyncPriorityQueue with capacity {capacity}, reuse_cooldown_time {replay_buffer.reuse_cooldown_time}."
            )
            return AsyncPriorityQueue(
                capacity=capacity,
                reuse_cooldown_time=replay_buffer.reuse_cooldown_time,
                priority_fn=replay_buffer.priority_fn,
                priority_fn_args=replay_buffer.priority_fn_args,
            )
        else:
            return AsyncQueue(capacity=config.capacity)


class AsyncQueue(asyncio.Queue, QueueBuffer):
    def __init__(self, capacity: int):
        """
        Initialize the async queue with a specified capacity.

        Args:
            capacity (`int`): The maximum number of items the queue can hold.
        """
        super().__init__(maxsize=capacity)
        self._closed = False
        self.min_model_version = 0

    async def put(self, item: List[Experience]):
        if len(item) == 0:
            return
        await super().put(item)

    async def get(self):
        while True:
            item = await super().get()
            if (
                self.min_model_version <= 0
                or item[0].info["model_version"] >= self.min_model_version
            ):
                return item

    async def close(self) -> None:
        """Close the queue."""
        self._closed = True
        getters = getattr(self, "_getters", [])
        for getter in getters:
            if not getter.done():
                getter.set_exception(StopAsyncIteration())
        getters.clear()

    def stopped(self) -> bool:
        """Check if there is no more data to read."""
        return self._closed and self.empty()


class AsyncPriorityQueue(QueueBuffer):
    """
    An asynchronous priority queue that manages a fixed-size buffer of experience items.
    Items are prioritized using a user-defined function and reinserted after a cooldown period.

    Attributes:
        capacity (int): Maximum number of items the queue can hold. This value is automatically
            adjusted to be at most twice the read batch size.
        reuse_cooldown_time (float): Delay before reusing an item (set to infinity to disable).
        priority_fn (callable): Function used to determine the priority of an item.
        priority_groups (SortedDict): Maps priorities to deques of items with the same priority.
    """

    def __init__(
        self,
        capacity: int,
        reuse_cooldown_time: Optional[float] = None,
        priority_fn: str = "linear_decay",
        priority_fn_args: Optional[dict] = None,
    ):
        """
        Initialize the async priority queue.

        Args:
            capacity (`int`): The maximum number of items the queue can store.
            reuse_cooldown_time (`float`): Time to wait before reusing an item. Set to None to disable reuse.
            priority_fn (`str`): Name of the function to use for determining item priority.
            kwargs: Additional keyword arguments for the priority function.
        """
        from trinity.buffer.storage import PRIORITY_FUNC

        self.capacity = capacity
        self.item_count = 0
        self.priority_groups = SortedDict()  # Maps priority -> deque of items
        priority_fn_cls = PRIORITY_FUNC.get(priority_fn)
        kwargs = priority_fn_cls.default_config()
        kwargs.update(priority_fn_args or {})
        self.priority_fn = priority_fn_cls(**kwargs)
        self.reuse_cooldown_time = reuse_cooldown_time
        self._condition = asyncio.Condition()  # For thread-safe operations
        self._closed = False
        self.min_model_version = 0

    async def _put(self, item: List[Experience], delay: float = 0) -> None:
        """
        Insert an item into the queue, replacing the lowest-priority item if full.

        Args:
            item (`List[Experience]`): A list of experiences to add.
            delay (`float`): Optional delay before insertion (for simulating timing behavior).
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if len(item) == 0:
            return

        priority, put_into_queue = self.priority_fn(item=item)
        if not put_into_queue:
            return

        async with self._condition:
            if self.item_count == self.capacity:
                # If full, only insert if new item has higher or equal priority than the lowest
                lowest_priority, item_queue = self.priority_groups.peekitem(index=0)
                if lowest_priority > priority:
                    return  # Skip insertion if lower priority
                # Remove the lowest priority item
                item_queue.popleft()
                self.item_count -= 1
                if not item_queue:
                    self.priority_groups.popitem(index=0)

            # Add the new item
            if priority not in self.priority_groups:
                self.priority_groups[priority] = deque()
            self.priority_groups[priority].append(item)
            self.item_count += 1
            self._condition.notify()

    async def put(self, item: List[Experience]) -> None:
        await self._put(item, delay=0)

    async def get(self) -> List[Experience]:
        """
        Retrieve the highest-priority item from the queue.

        Returns:
            List[Experience]: The highest-priority item (list of experiences).

        Notes:
            - After retrieval, the item is optionally reinserted after a cooldown period.
        """
        async with self._condition:
            while True:
                while len(self.priority_groups) == 0:
                    if self._closed:
                        raise StopAsyncIteration()
                    await self._condition.wait()

                _, item_queue = self.priority_groups.peekitem(index=-1)
                item = item_queue.popleft()
                self.item_count -= 1
                if not item_queue:
                    self.priority_groups.popitem(index=-1)

                if (
                    self.min_model_version <= 0
                    or item[0].info["model_version"] >= self.min_model_version
                ):
                    break

        for exp in item:
            exp.info["use_count"] += 1
        # Optionally resubmit the item after a cooldown
        if self.reuse_cooldown_time is not None:
            asyncio.create_task(self._put(item, delay=self.reuse_cooldown_time))

        return item

    def qsize(self):
        return self.item_count

    async def close(self) -> None:
        """
        Close the queue.
        """
        async with self._condition:
            self._closed = True
            # No more items will be added, but existing items can still be processed.
            self.reuse_cooldown_time = None
            self._condition.notify_all()

    def stopped(self) -> bool:
        return self._closed and len(self.priority_groups) == 0


class QueueStorage:
    """An wrapper of a async queue."""

    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"queue_{config.name}", in_ray_actor=True)
        self.config = config
        self.capacity = config.capacity
        self.queue = QueueBuffer.get_queue(config)
        st_config = deepcopy(config)
        st_config.wrap_in_ray = False
        if st_config.path:
            if is_database_url(st_config.path):
                from trinity.buffer.writer.sql_writer import SQLWriter

                st_config.storage_type = StorageType.SQL.value
                self.writer = SQLWriter(st_config)
            elif is_json_file(st_config.path):
                from trinity.buffer.writer.file_writer import JSONWriter

                st_config.storage_type = StorageType.FILE.value
                self.writer = JSONWriter(st_config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            from trinity.buffer.writer.file_writer import JSONWriter

            st_config.storage_type = StorageType.FILE.value
            self.writer = JSONWriter(st_config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0
        self.exp_pool = deque()  # A pool to store experiences
        self.closed = False

        self.zmq_config = config.zmq
        self._zmq_enabled = bool(self.zmq_config and self.zmq_config.enable)
        self._zmq_context = None
        self._zmq_pull_socket = None
        self._zmq_rep_socket = None
        self._zmq_server_task = None
        self._zmq_server_lock = asyncio.Lock()
        self._zmq_endpoints: Dict[str, str] = {}

        if self._zmq_enabled:
            self.logger.warning("QueueStorage ZeroMQ data transport is enabled.")

    async def _ensure_zmq_server(self) -> None:
        if not self._zmq_enabled:
            return
        zmq_config = self.zmq_config
        if zmq_config is None:
            return
        async with self._zmq_server_lock:
            if self._zmq_server_task is not None and not self._zmq_server_task.done():
                return

            try:
                import zmq
                import zmq.asyncio
            except ImportError as exc:
                raise RuntimeError(
                    "ZeroMQ transport is enabled, but dependency `pyzmq` is not installed."
                ) from exc

            self._zmq_context = zmq.asyncio.Context.instance()
            self._zmq_pull_socket = self._zmq_context.socket(zmq.PULL)
            self._zmq_rep_socket = self._zmq_context.socket(zmq.REP)

            self._zmq_pull_socket.setsockopt(zmq.RCVHWM, zmq_config.rcvhwm)
            self._zmq_pull_socket.setsockopt(zmq.LINGER, zmq_config.linger_ms)
            self._zmq_rep_socket.setsockopt(zmq.SNDHWM, zmq_config.sndhwm)
            self._zmq_rep_socket.setsockopt(zmq.RCVHWM, zmq_config.rcvhwm)
            self._zmq_rep_socket.setsockopt(zmq.LINGER, zmq_config.linger_ms)

            bind_host = zmq_config.bind_host
            writer_port = zmq_config.writer_port
            reader_port = zmq_config.reader_port

            if writer_port > 0:
                self._zmq_pull_socket.bind(f"tcp://{bind_host}:{writer_port}")
            else:
                writer_port = self._zmq_pull_socket.bind_to_random_port(f"tcp://{bind_host}")

            if reader_port > 0:
                self._zmq_rep_socket.bind(f"tcp://{bind_host}:{reader_port}")
            else:
                reader_port = self._zmq_rep_socket.bind_to_random_port(f"tcp://{bind_host}")

            connect_host = zmq_config.connect_host or ray.util.get_node_ip_address()
            self._zmq_endpoints = {
                "writer_endpoint": f"tcp://{connect_host}:{writer_port}",
                "reader_endpoint": f"tcp://{connect_host}:{reader_port}",
            }

            self._zmq_server_task = asyncio.create_task(self._zmq_server_loop())
            self.logger.warning(
                "ZeroMQ server started for queue %s, writer=%s, reader=%s",
                self.config.name,
                self._zmq_endpoints["writer_endpoint"],
                self._zmq_endpoints["reader_endpoint"],
            )

    async def _stop_zmq_server(self) -> None:
        task = self._zmq_server_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.warning("Error when stopping ZeroMQ server: %s", e)
        self._zmq_server_task = None

        if self._zmq_pull_socket is not None:
            self._zmq_pull_socket.close(0)
            self._zmq_pull_socket = None
        if self._zmq_rep_socket is not None:
            self._zmq_rep_socket.close(0)
            self._zmq_rep_socket = None
        self._zmq_endpoints = {}

    async def _zmq_server_loop(self) -> None:
        import zmq
        import zmq.asyncio

        if self._zmq_pull_socket is None or self._zmq_rep_socket is None:
            return

        poller = zmq.asyncio.Poller()
        poller.register(self._zmq_pull_socket, zmq.POLLIN)
        poller.register(self._zmq_rep_socket, zmq.POLLIN)

        try:
            while True:
                events = dict(await poller.poll(timeout=1000))

                if self._zmq_pull_socket in events:
                    payload = await self._zmq_pull_socket.recv()
                    exps = Experience.deserialize_many(payload)
                    await self.put_batch(exps)

                if self._zmq_rep_socket in events:
                    request = await self._zmq_rep_socket.recv_json()
                    command = request.get("cmd", "get_batch")

                    if command == "ping":
                        await self._zmq_rep_socket.send_multipart([b"ok", b"pong"])
                        continue

                    if command != "get_batch":
                        await self._zmq_rep_socket.send_multipart(
                            [b"error", f"Unknown command: {command}".encode("utf-8")]
                        )
                        continue

                    batch_size = int(request.get("batch_size", self.config.batch_size or 1))
                    timeout_sec = float(request.get("timeout", self.config.max_read_timeout))
                    min_model_version = int(request.get("min_model_version", 0))
                    try:
                        exps = await self.get_batch(
                            batch_size=batch_size,
                            timeout=timeout_sec,
                            min_model_version=min_model_version,
                        )
                        payload = Experience.serialize_many(exps)
                        await self._zmq_rep_socket.send_multipart([b"ok", payload])
                    except StopAsyncIteration:
                        await self._zmq_rep_socket.send_multipart([b"eos", b""])
                    except Exception as e:
                        await self._zmq_rep_socket.send_multipart([b"error", str(e).encode("utf-8")])
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.exception("ZeroMQ server loop crashed: %s", e)

    async def get_zmq_endpoints(self) -> Dict[str, Any]:
        if not self._zmq_enabled:
            return {"enabled": False}
        zmq_config = self.zmq_config
        if zmq_config is None:
            return {"enabled": False}

        await self._ensure_zmq_server()
        return {
            "enabled": True,
            "writer_endpoint": self._zmq_endpoints["writer_endpoint"],
            "reader_endpoint": self._zmq_endpoints["reader_endpoint"],
        }

    async def acquire(self) -> int:
        if self._zmq_enabled:
            await self._ensure_zmq_server()
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.closed = True
            if self._zmq_enabled:
                await self._stop_zmq_server()
            await self.queue.close()
            if self.writer is not None:
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

    async def get_batch(self, batch_size: int, timeout: float, min_model_version: int = 0) -> List:
        """Get batch of experience."""
        await self.queue.set_min_model_version(min_model_version)
        start_time = time.time()
        result = []
        while len(result) < batch_size:
            while len(self.exp_pool) > 0 and len(result) < batch_size:
                exp = self.exp_pool.popleft()
                if min_model_version > 0 and exp.info["model_version"] < min_model_version:
                    continue
                result.append(exp)
            if len(result) >= batch_size:
                break

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
        return result

    @classmethod
    def get_wrapper(cls, config: StorageConfig):
        """Get the queue actor."""
        return (
            ray.remote(cls)
            .options(
                name=f"queue-{config.name}",
                namespace=config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
            )
            .remote(config)
        )
