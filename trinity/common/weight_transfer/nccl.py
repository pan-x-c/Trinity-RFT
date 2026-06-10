# -*- coding: utf-8 -*-
"""NCCL-based weight transfer with ZMQ metadata side-channel.

Provides three classes for weight synchronization between trainer and
rollout/inference workers:

* :class:`ModelWeightSender` — trainer-side sender supporting two modes:

  - **Bucket mode** (default): double-buffered GPU buckets over NCCL
    broadcast, overlapping communication with bucket filling.
  - **Per-tensor mode** (``per_tensor=True``): individual NCCL broadcasts
    per tensor with ZMQ-batched metadata.  No GPU buffers are allocated.
    Designed for backends (e.g. SGLang) that manage NCCL reception
    internally via HTTP API.

* :class:`ModelWeightReceiver` — rollout-side receiver with GPU double
  buffers.  Pairs with the Sender's bucket mode.

* :class:`ModelWeightMetadataReceiver` — lightweight metadata-only receiver
  (ZMQ SUB, no GPU).  Pairs with the Sender's per-tensor mode for backends
  that handle NCCL reception internally.

Usage (bucket mode)::

    # --- Trainer (rank 0) ---
    sender = ModelWeightSender()
    sender.setup(pg, bucket_size=500_000_000)
    await sender.send(engine.get_per_tensor_param()[0])
    sender.finalize()

    # --- Rollout (rank 1+) ---
    receiver = ModelWeightReceiver(pg, 500_000_000, zmq_ip, zmq_port)
    async for name, tensor in receiver.receive():
        model.load_weight(name, tensor)
    receiver.finalize()

Usage (per-tensor mode, e.g. SGLang)::

    # --- Trainer (rank 0) ---
    sender = ModelWeightSender()
    sender.setup(pg, bucket_size=500_000_000, per_tensor=True)
    sender.send_sync(engine.get_per_tensor_param()[0])
    sender.finalize()

    # --- SGLang control plane ---
    meta_receiver = ModelWeightMetadataReceiver(zmq_ip, zmq_port)
    for batch, is_last in meta_receiver.receive():
        api_client.update_weights_from_distributed(batch)
    meta_receiver.finalize()
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import AsyncGenerator, Dict, Iterable, Iterator, Optional, Tuple

import torch
import torch.distributed
import zmq

from trinity.common.weight_transfer.core import (
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)


def _get_local_ip() -> str:
    """Get the node's IP address, preferring Ray if available."""
    try:
        import ray

        return ray.util.get_node_ip_address().strip("[]")
    except Exception:
        import socket

        return socket.gethostbyname(socket.gethostname())


def _get_free_port(ip: str) -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((ip, 0))
        return s.getsockname()[1]


def _is_ipv6(ip: str) -> bool:
    """Check if an IP address is IPv6."""
    import ipaddress

    try:
        return isinstance(ipaddress.ip_address(ip), ipaddress.IPv6Address)
    except ValueError:
        return False


class _BroadcastFuture:
    """Wraps a broadcast operation running in a background thread.

    Handles both ZMQ metadata exchange and NCCL broadcast in a single
    thread so the main coroutine can fill the next bucket concurrently.
    """

    def __init__(
        self,
        is_sender: bool,
        pg: torch.distributed.ProcessGroup,
        bucket: torch.Tensor,
        metadata: Optional[Dict],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self._is_sender = is_sender
        self._pg = pg
        self._bucket = bucket
        self._metadata = metadata
        self._socket = socket
        self._topic = topic

        self._loop = asyncio.get_running_loop()
        self._future = self._loop.run_in_executor(None, self._run)

    def _run(self):
        """Execute ZMQ metadata exchange + NCCL broadcast (runs in thread)."""
        if self._is_sender:
            self._socket.send_string(self._topic, flags=zmq.SNDMORE)
            self._socket.send_pyobj(self._metadata)
        else:
            self._socket.recv_string()
            self._metadata = self._socket.recv_pyobj()

        torch.distributed.broadcast(self._bucket, src=0, group=self._pg)

    async def wait(self) -> Optional[Dict]:
        """Wait for the broadcast to complete, return received metadata."""
        await self._future
        return self._metadata


class ModelWeightSender:
    """Sends model weights via double-buffered NCCL broadcast.

    Sits on the trainer side (NCCL rank 0).  Accepts weight tensors from
    any source (FSDP ``get_per_tensor_param``, plain ``state_dict``, etc.)
    and broadcasts them to all receivers in fixed-size buckets.

    Two-phase lifecycle:

    1. ``__init__()`` — lightweight: only binds a ZMQ PUB socket for
       the metadata side-channel.  No GPU memory is allocated.
    2. ``setup(pg, bucket_size)`` — heavy: allocates GPU double buffers
       and wires the NCCL process group.

    This split exists because the ZMQ port must be known *before* the
    concurrent NCCL ``init_process_group`` (the explorer needs it), while
    the process group is only available *after* that collective call.
    """

    def __init__(self) -> None:
        self.bucket_size: int = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pg: Optional[torch.distributed.ProcessGroup] = None
        self._send_buf: Optional[torch.Tensor] = None
        self._recv_buf: Optional[torch.Tensor] = None
        self._per_tensor: bool = False
        self._topic = "bucket_metadata"

        # Bind ZMQ PUB server for metadata broadcast.
        self._zmq_ip = _get_local_ip()
        self._zmq_port = _get_free_port(self._zmq_ip)
        self._zmq_context: zmq.Context = zmq.Context()
        self._socket: zmq.Socket = self._zmq_context.socket(zmq.PUB)

        if _is_ipv6(self._zmq_ip):
            address = f"tcp://[{self._zmq_ip}]:{self._zmq_port}"
            self._socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self._zmq_ip}:{self._zmq_port}"

        self._socket.bind(address)
        self.logger.info(f"ModelWeightSender ZMQ PUB bound to {address}")

    @property
    def zmq_info(self) -> Dict[str, object]:
        """ZMQ connection info for receivers: ``{zmq_ip, zmq_port}``."""
        return {"zmq_ip": self._zmq_ip, "zmq_port": self._zmq_port}

    def setup(
        self,
        pg: torch.distributed.ProcessGroup,
        bucket_size: int,
        per_tensor: bool = False,
    ) -> None:
        """Wire the NCCL process group and configure the transfer mode.

        Must be called after the NCCL process group has been created
        (via ``init_process_group``).

        Args:
            pg: The NCCL process group connecting trainer rank 0 to all
                rollout workers.
            bucket_size: In **bucket mode**, each GPU buffer size in bytes
                (two buffers allocated, total overhead = ``2 * bucket_size``).
                In **per-tensor mode**, the byte threshold for batching
                ZMQ metadata messages (no GPU buffers allocated).
            per_tensor: When ``True``, use per-tensor NCCL broadcasts
                with ZMQ metadata batching instead of GPU double buffers.
                Designed for backends (e.g. SGLang) that manage NCCL
                reception internally.
        """
        self.bucket_size = bucket_size
        self._pg = pg
        self._per_tensor = per_tensor
        if not per_tensor:
            self._send_buf = torch.empty(bucket_size, dtype=torch.uint8, device="cuda")
            self._recv_buf = torch.empty(bucket_size, dtype=torch.uint8, device="cuda")

    @torch.no_grad()
    async def send(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Send all weights via NCCL broadcast.

        Dispatches to bucketed or per-tensor mode based on the
        ``per_tensor`` flag set during :meth:`setup`.

        Args:
            weights: Iterable of ``(name, tensor)`` pairs — typically from
                ``engine.get_per_tensor_param()`` or a ``state_dict.items()``.
        """
        assert self._pg is not None, "Call setup() first"
        if self._per_tensor:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_per_tensor_impl, weights)
        else:
            await self._send_bucketed(weights)

    @torch.no_grad()
    async def _send_bucketed(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Send all weights via double-buffered NCCL broadcast (bucket mode).

        Iterates over *weights*, packs tensor chunks into fixed-size
        buckets, and broadcasts each bucket.  While one bucket is being
        broadcast (in a background thread), the next bucket is filled
        concurrently.
        """
        assert self._send_buf is not None, "Call setup() first"

        send_buf = self._send_buf
        recv_buf = self._recv_buf
        broadcast_op: Optional[_BroadcastFuture] = None

        start_time = time.time()
        bucket_meta: Dict[str, TensorMeta] = {}
        offset = 0

        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # If chunk doesn't fit in current bucket, flush it.
            if offset + tensor_meta.chunk_size > self.bucket_size:
                torch.cuda.synchronize()

                # Wait for the previous broadcast to finish.
                if broadcast_op is not None:
                    await broadcast_op.wait()

                # Launch broadcast for the current (full) bucket.
                broadcast_op = _BroadcastFuture(
                    is_sender=True,
                    pg=self._pg,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self._socket,
                    topic=self._topic,
                )

                # Swap buffers: fill the other one while this broadcasts.
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size].copy_(chunk)
            offset += tensor_meta.chunk_size

        # Flush the final (possibly partial) bucket.
        torch.cuda.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait()

        broadcast_op = _BroadcastFuture(
            is_sender=True,
            pg=self._pg,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self._socket,
            topic=self._topic,
        )
        await broadcast_op.wait()

        elapsed = time.time() - start_time
        self.logger.info(f"ModelWeightSender: send completed in {elapsed:.2f}s")

    # -- Per-tensor mode ---------------------------------------------------

    @torch.no_grad()
    def _send_per_tensor_impl(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Per-tensor send: ZMQ metadata batching + individual NCCL broadcasts.

        Accumulates tensors until the total byte count reaches
        ``bucket_size``, then sends a single ZMQ metadata message
        followed by individual ``broadcast`` calls for each tensor
        in the batch.  Tensors are ``.clone()``-d to prevent dangling
        references when the source is an FSDP ``summon_full_params``
        generator.
        """
        start_time = time.time()
        batch_meta: list = []
        batch_tensors: list = []
        batch_bytes = 0
        total_tensors = 0

        for name, tensor in weights:
            dtype_str = str(tensor.dtype).replace("torch.", "")
            shape = tuple(tensor.shape)
            batch_meta.append((name, dtype_str, shape))
            batch_tensors.append(tensor.detach().clone())
            batch_bytes += tensor.nbytes

            if batch_bytes >= self.bucket_size:
                self._flush_per_tensor_batch(batch_meta, batch_tensors, is_last=False)
                total_tensors += len(batch_tensors)
                batch_meta, batch_tensors, batch_bytes = [], [], 0

        # Final batch (always sent, even if empty, to deliver is_last=True).
        self._flush_per_tensor_batch(batch_meta, batch_tensors, is_last=True)
        total_tensors += len(batch_tensors)

        elapsed = time.time() - start_time
        self.logger.info(
            f"ModelWeightSender: per-tensor send completed in "
            f"{elapsed:.2f}s ({total_tensors} tensors)"
        )

    def _flush_per_tensor_batch(
        self,
        batch_meta: list,
        batch_tensors: list,
        is_last: bool,
    ) -> None:
        """Send ZMQ metadata message, then broadcast each tensor individually."""
        self._socket.send_string(self._topic, flags=zmq.SNDMORE)
        self._socket.send_pyobj({"tensor_meta": batch_meta, "is_last": is_last})
        for tensor in batch_tensors:
            torch.distributed.broadcast(tensor.contiguous(), src=0, group=self._pg)

    @torch.no_grad()
    def send_sync(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Synchronous wrapper around :meth:`send`.

        In **per-tensor mode**, calls the synchronous implementation
        directly.  In **bucket mode**, runs the async send in a
        dedicated event loop.  Safe to call from synchronous contexts
        (e.g. vLLM ``collective_rpc`` handlers).
        """
        if self._per_tensor:
            self._send_per_tensor_impl(weights)
        else:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.send(weights))
            finally:
                loop.close()

    def finalize(self) -> None:
        """Release GPU buffers and close the ZMQ socket."""
        self._send_buf = None
        self._recv_buf = None

        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        torch.cuda.empty_cache()


class ModelWeightReceiver:
    """Receives model weights via double-buffered NCCL broadcast.

    Sits on the rollout/inference side (NCCL rank 1+).  Receives buckets
    from :class:`ModelWeightSender` and yields individual weight tensors.

    All initialization is done in ``__init__``: GPU buffer allocation,
    NCCL process group wiring, and ZMQ SUB connection.

    Args:
        pg: The NCCL process group connecting trainer rank 0 to all
            rollout workers.
        bucket_size: Must match the sender's ``bucket_size`` (in bytes).
        zmq_ip: The sender's IP address (from ``sender.zmq_info``).
        zmq_port: The sender's ZMQ port (from ``sender.zmq_info``).
    """

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        bucket_size: int,
        zmq_ip: str,
        zmq_port: int,
    ) -> None:
        self.bucket_size = bucket_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pg = pg
        self._topic = "bucket_metadata"

        # Allocate GPU double buffers.
        self._send_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cuda")
        self._recv_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cuda")

        # Connect ZMQ SUB socket to the sender's PUB server.
        self._zmq_context: zmq.Context = zmq.Context()
        self._socket: zmq.Socket = self._zmq_context.socket(zmq.SUB)

        if _is_ipv6(zmq_ip):
            address = f"tcp://[{zmq_ip}]:{zmq_port}"
            self._socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{zmq_ip}:{zmq_port}"

        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self.logger.info(f"ModelWeightReceiver ZMQ SUB connected to {address}")

    @torch.no_grad()
    async def receive(self) -> AsyncGenerator[Tuple[str, torch.Tensor], None]:
        """Receive all weights via double-buffered NCCL broadcast.

        Yields ``(name, tensor)`` pairs as they are fully received and
        reassembled.  Large tensors split across multiple buckets are
        automatically merged.

        The method overlaps NCCL reception of the next bucket with
        consumption (yielding) of the current bucket's tensors.
        """
        async for name, weight in merge_weight_chunks(self._receive_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_chunks(
        self,
    ) -> AsyncGenerator[Tuple[TensorMeta, torch.Tensor], None]:
        """Receive raw bucket chunks with double-buffering."""
        assert self._pg is not None
        assert self._recv_buf is not None

        send_buf = self._send_buf
        recv_buf = self._recv_buf
        total_bytes = 0
        total_params = 0

        start_time = time.time()

        # Receive the first bucket.
        broadcast_op = _BroadcastFuture(
            is_sender=False,
            pg=self._pg,
            bucket=recv_buf,
            metadata=None,
            socket=self._socket,
            topic=self._topic,
        )
        metadata = await broadcast_op.wait()
        assert metadata is not None, "Receiver must get metadata from sender"
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # Swap: recv_buf (now filled) becomes send_buf (to yield from).
        send_buf, recv_buf = recv_buf, send_buf

        while not metadata["is_last"]:
            # 1. Start receiving next bucket in background.
            broadcast_op = _BroadcastFuture(
                is_sender=False,
                pg=self._pg,
                bucket=recv_buf,
                metadata=None,
                socket=self._socket,
                topic=self._topic,
            )

            # 2. Yield tensors from the completed bucket (send_buf).
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. Wait for next bucket.
            metadata = await broadcast_op.wait()
            assert metadata is not None, "Receiver must get metadata from sender"
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. Swap and sync.
            torch.cuda.synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        # Yield tensors from the final bucket.
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024) if elapsed > 0 else 0
        self.logger.debug(
            f"ModelWeightReceiver: received {total_params} params in "
            f"{elapsed:.2f}s ({bandwidth:.2f} GB/s)"
        )

    def receive_sync(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Synchronous wrapper around :meth:`receive`.

        Runs the async receive loop in a dedicated thread with its own
        event loop, feeding items into a :class:`queue.Queue` that the
        returned iterator drains.  This is safe to call from synchronous
        contexts (e.g. vLLM's ``reload_weights``).

        Yields:
            ``(name, tensor)`` pairs identical to :meth:`receive`.
        """
        q: queue.Queue = queue.Queue()
        sentinel = object()
        error_holder: list = []

        def _run():
            loop = asyncio.new_event_loop()
            try:

                async def _drain():
                    async for item in self.receive():
                        q.put(item)
                    q.put(sentinel)

                loop.run_until_complete(_drain())
            except Exception as e:
                error_holder.append(e)
                q.put(sentinel)
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

        t.join()

        if error_holder:
            raise error_holder[0]

    def finalize(self) -> None:
        """Release GPU buffers and close the ZMQ socket."""
        self._send_buf = None
        self._recv_buf = None

        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        torch.cuda.empty_cache()


class ModelWeightMetadataReceiver:
    """Receives weight metadata via ZMQ for per-tensor NCCL broadcast.

    Simplified counterpart of :class:`ModelWeightReceiver` for backends
    (e.g. SGLang) that manage NCCL reception internally.  Only subscribes
    to the :class:`ModelWeightSender`'s ZMQ PUB socket for batched tensor
    metadata; no GPU buffers are allocated and no NCCL operations are
    performed.

    Pairs with :class:`ModelWeightSender` in per-tensor mode
    (``setup(pg, bucket_size, per_tensor=True)``).

    Args:
        zmq_ip: The sender's IP address (from ``sender.zmq_info``).
        zmq_port: The sender's ZMQ port (from ``sender.zmq_info``).
    """

    def __init__(self, zmq_ip: str, zmq_port: int) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._topic = "bucket_metadata"

        # Connect ZMQ SUB socket to the sender's PUB server.
        self._zmq_context: zmq.Context = zmq.Context()
        self._socket: zmq.Socket = self._zmq_context.socket(zmq.SUB)

        if _is_ipv6(zmq_ip):
            address = f"tcp://[{zmq_ip}]:{zmq_port}"
            self._socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{zmq_ip}:{zmq_port}"

        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self.logger.info(f"ModelWeightMetadataReceiver ZMQ SUB connected to {address}")

    def receive(self) -> Iterator[Tuple[list, bool]]:
        """Yield ``(batch_meta, is_last)`` pairs from the Sender.

        Each ``batch_meta`` is a list of ``(name, dtype_str, shape)``
        tuples describing the tensors that will be broadcast in the
        next round of per-tensor NCCL broadcasts.

        The caller should trigger backend-specific NCCL reception
        (e.g. via HTTP API) for each batch.

        Yields:
            ``(batch_meta, is_last)`` where *batch_meta* is
            ``List[Tuple[str, str, Tuple]]`` and *is_last* is ``bool``.
        """
        while True:
            self._socket.recv_string()  # topic filter
            metadata = self._socket.recv_pyobj()
            tensor_meta = metadata["tensor_meta"]
            is_last = metadata["is_last"]
            yield tensor_meta, is_last
            if is_last:
                break

    def finalize(self) -> None:
        """Close the ZMQ socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None
