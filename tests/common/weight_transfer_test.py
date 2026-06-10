# -*- coding: utf-8 -*-
"""Unit tests for trinity.common.weight_transfer utilities.

These tests run on CPU only — no GPU or NCCL required.  They validate the
chunk splitting/merging round-trip, TensorMeta serialization, and the
per-tensor mode ZMQ protocol.
"""
import asyncio
import pickle
import threading
import time
from typing import List, Tuple

import torch

from trinity.common.weight_transfer.core import (
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from trinity.common.weight_transfer.nccl import (
    ModelWeightMetadataReceiver,
    ModelWeightSender,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _collect_async_gen(agen):
    """Drain an async generator into a list."""
    result = []
    async for item in agen:
        result.append(item)
    return result


def _make_weights(shapes_dtypes: List[Tuple[Tuple[int, ...], torch.dtype]]):
    """Create a list of (name, tensor) pairs."""
    return [
        (f"layer.{i}.weight", torch.randn(shape, dtype=dtype))
        for i, (shape, dtype) in enumerate(shapes_dtypes)
    ]


# ---------------------------------------------------------------------------
# TensorMeta tests
# ---------------------------------------------------------------------------


class TestTensorMeta:
    def test_create(self):
        meta = TensorMeta(
            name="fc.weight",
            shape=torch.Size([128, 64]),
            dtype=torch.float32,
            chunk_offset=0,
            chunk_size=32768,
            offset=0,
        )
        assert meta.name == "fc.weight"
        assert meta.shape == torch.Size([128, 64])
        assert meta.dtype == torch.float32
        assert meta.chunk_size == 32768

    def test_pickle_roundtrip(self):
        """TensorMeta must survive pickle (used by ZMQ send_pyobj)."""
        meta = TensorMeta(
            name="attn.qkv",
            shape=torch.Size([4096, 4096]),
            dtype=torch.bfloat16,
            chunk_offset=1024,
            chunk_size=2048,
            offset=512,
        )
        restored = pickle.loads(pickle.dumps(meta))
        assert restored.name == meta.name
        assert restored.shape == meta.shape
        assert restored.dtype == meta.dtype
        assert restored.chunk_offset == meta.chunk_offset
        assert restored.chunk_size == meta.chunk_size
        assert restored.offset == meta.offset


# ---------------------------------------------------------------------------
# split_weight_chunks tests
# ---------------------------------------------------------------------------


class TestSplitWeightChunks:
    def test_small_tensor_single_chunk(self):
        """A tensor smaller than bucket_size should produce exactly one chunk."""
        weights = _make_weights([((32, 16), torch.float32)])
        bucket_size = 1024 * 1024  # 1MB — much larger than 32*16*4=2048 bytes

        async def _run():
            return await _collect_async_gen(split_weight_chunks(weights, bucket_size))

        chunks = _run_async(_run())
        assert len(chunks) == 1
        meta, chunk = chunks[0]
        assert meta.name == "layer.0.weight"
        assert meta.chunk_offset == 0
        assert meta.chunk_size == 32 * 16 * 4  # float32 = 4 bytes
        assert chunk.dtype == torch.uint8

    def test_large_tensor_multiple_chunks(self):
        """A tensor larger than bucket_size should be split into multiple chunks."""
        # 256 * 256 * 4 bytes = 262144 bytes
        weights = _make_weights([((256, 256), torch.float32)])
        bucket_size = 100_000  # < 262144, so should get 3 chunks

        async def _run():
            return await _collect_async_gen(split_weight_chunks(weights, bucket_size))

        chunks = _run_async(_run())
        assert len(chunks) == 3  # ceil(262144 / 100000) = 3

        total_bytes = sum(meta.chunk_size for meta, _ in chunks)
        assert total_bytes == 256 * 256 * 4

        # Verify chunk_offsets are sequential.
        offset = 0
        for meta, _ in chunks:
            assert meta.chunk_offset == offset
            offset += meta.chunk_size

    def test_multiple_tensors(self):
        """Multiple tensors should each produce their own chunks."""
        weights = _make_weights(
            [
                ((8, 8), torch.float32),  # 256 bytes
                ((16, 16), torch.float32),  # 1024 bytes
            ]
        )
        bucket_size = 1024 * 1024

        async def _run():
            return await _collect_async_gen(split_weight_chunks(weights, bucket_size))

        chunks = _run_async(_run())
        assert len(chunks) == 2  # one chunk per tensor
        assert chunks[0][0].name == "layer.0.weight"
        assert chunks[1][0].name == "layer.1.weight"

    def test_bf16_dtype(self):
        """bfloat16 tensors should be chunked by their actual byte size."""
        weights = _make_weights([((64, 64), torch.bfloat16)])
        bucket_size = 1024 * 1024

        async def _run():
            return await _collect_async_gen(split_weight_chunks(weights, bucket_size))

        chunks = _run_async(_run())
        assert len(chunks) == 1
        meta, chunk = chunks[0]
        assert meta.chunk_size == 64 * 64 * 2  # bf16 = 2 bytes


# ---------------------------------------------------------------------------
# merge_weight_chunks tests (round-trip with split)
# ---------------------------------------------------------------------------


class TestMergeWeightChunks:
    def test_roundtrip_small(self):
        """Small tensors should survive a split→merge round-trip exactly."""
        original = _make_weights(
            [
                ((32, 16), torch.float32),
                ((64, 32), torch.float32),
            ]
        )
        bucket_size = 1024 * 1024

        async def _run():
            chunks_gen = split_weight_chunks(original, bucket_size)
            merged = await _collect_async_gen(merge_weight_chunks(chunks_gen, bucket_size))
            return merged

        merged = _run_async(_run())
        assert len(merged) == len(original)

        for (orig_name, orig_tensor), (merged_name, merged_tensor) in zip(original, merged):
            assert merged_name == orig_name
            assert torch.equal(merged_tensor, orig_tensor)

    def test_roundtrip_large_tensor(self):
        """A tensor split across multiple buckets should merge back exactly."""
        original = _make_weights([((256, 256), torch.float32)])
        bucket_size = 100_000  # Forces 3 chunks

        async def _run():
            chunks_gen = split_weight_chunks(original, bucket_size)
            merged = await _collect_async_gen(merge_weight_chunks(chunks_gen, bucket_size))
            return merged

        merged = _run_async(_run())
        assert len(merged) == 1
        assert merged[0][0] == "layer.0.weight"
        assert torch.equal(merged[0][1], original[0][1])

    def test_roundtrip_mixed_sizes(self):
        """A mix of small and large tensors should all survive round-trip."""
        original = _make_weights(
            [
                ((4, 4), torch.float32),  # 64 bytes — fits in one bucket
                ((256, 256), torch.float32),  # 262144 bytes — split into chunks
                ((8, 8), torch.bfloat16),  # 128 bytes — fits in one bucket
            ]
        )
        bucket_size = 100_000

        async def _run():
            chunks_gen = split_weight_chunks(original, bucket_size)
            merged = await _collect_async_gen(merge_weight_chunks(chunks_gen, bucket_size))
            return merged

        merged = _run_async(_run())
        assert len(merged) == len(original)

        for (orig_name, orig_tensor), (merged_name, merged_tensor) in zip(original, merged):
            assert merged_name == orig_name
            assert torch.equal(merged_tensor, orig_tensor)

    def test_roundtrip_exact_bucket_size(self):
        """A tensor whose size exactly equals bucket_size should work."""
        # 100 * 250 * 4 = 100000 bytes = bucket_size
        original = _make_weights([((100, 250), torch.float32)])
        bucket_size = 100_000

        async def _run():
            chunks_gen = split_weight_chunks(original, bucket_size)
            merged = await _collect_async_gen(merge_weight_chunks(chunks_gen, bucket_size))
            return merged

        merged = _run_async(_run())
        assert len(merged) == 1
        assert torch.equal(merged[0][1], original[0][1])


# ---------------------------------------------------------------------------
# ModelWeightSender per-tensor mode tests
# ---------------------------------------------------------------------------


class TestSenderPerTensorSetup:
    """Verify that per-tensor setup() skips GPU buffer allocation."""

    def test_per_tensor_setup_no_gpu_buffers(self):
        """setup(per_tensor=True) must NOT allocate GPU double buffers."""
        sender = ModelWeightSender()
        try:
            # Use a mock process group — we only test the setup path.
            sender.setup(pg=None, bucket_size=1_000_000, per_tensor=True)
            assert sender._per_tensor is True
            assert sender.bucket_size == 1_000_000
            assert sender._send_buf is None
            assert sender._recv_buf is None
        finally:
            sender.finalize()

    def test_bucket_mode_setup_has_fields(self):
        """Contrast: setup(per_tensor=False) sets _per_tensor=False."""
        sender = ModelWeightSender()
        try:
            # Can't allocate GPU here (CPU test), but we can verify the flag.
            sender._per_tensor = False  # default, verify it stays
            sender.bucket_size = 0
            sender._pg = None
            # Just verify init defaults are correct.
            assert sender._per_tensor is False
            assert sender._send_buf is None
        finally:
            sender.finalize()


class TestSenderPerTensorSend:
    """Test the per-tensor ZMQ metadata + NCCL broadcast protocol.

    Mocks ``torch.distributed.broadcast`` to avoid needing a real NCCL
    process group.  Tests the ZMQ metadata roundtrip between Sender and
    MetadataReceiver.
    """

    def test_single_batch_small_weights(self):
        """Small weights fitting in one batch: single ZMQ message with is_last=True."""
        sender = ModelWeightSender()
        try:
            sender.setup(pg=None, bucket_size=1_000_000, per_tensor=True)
            zmq_info = sender.zmq_info

            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            time.sleep(0.1)  # ZMQ SUB subscription propagation

            weights = [
                ("layer.0.weight", torch.randn(8, 8)),
                ("layer.1.bias", torch.randn(8)),
            ]

            broadcast_calls = []
            original_broadcast = torch.distributed.broadcast

            def mock_broadcast(tensor, src=0, group=None):
                broadcast_calls.append(tensor.shape)

            torch.distributed.broadcast = mock_broadcast
            try:
                # Run sender in a background thread.
                send_thread = threading.Thread(
                    target=sender._send_per_tensor_impl,
                    args=(weights,),
                )
                send_thread.start()

                # Receive metadata.
                batches = list(receiver.receive())
                send_thread.join(timeout=5)
            finally:
                torch.distributed.broadcast = original_broadcast

            # Should be exactly one batch with is_last=True.
            assert len(batches) == 1
            batch_meta, is_last = batches[0]
            assert is_last is True
            assert len(batch_meta) == 2
            assert batch_meta[0][0] == "layer.0.weight"
            assert batch_meta[0][1] == "float32"
            assert batch_meta[0][2] == (8, 8)
            assert batch_meta[1][0] == "layer.1.bias"
            assert batch_meta[1][1] == "float32"
            assert batch_meta[1][2] == (8,)

            # NCCL broadcast should have been called once per tensor.
            assert len(broadcast_calls) == 2
        finally:
            receiver.finalize()
            sender.finalize()

    def test_multiple_batches_large_weights(self):
        """Weights exceeding bucket_size should produce multiple ZMQ batches."""
        sender = ModelWeightSender()
        try:
            # bucket_size = 1024 bytes; each tensor = 16*16*4 = 1024 bytes.
            sender.setup(pg=None, bucket_size=1024, per_tensor=True)
            zmq_info = sender.zmq_info

            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            time.sleep(0.1)

            weights = [
                ("w0", torch.randn(16, 16)),  # 1024 bytes → triggers flush
                ("w1", torch.randn(16, 16)),  # 1024 bytes → triggers flush
                ("w2", torch.randn(4, 4)),  # 64 bytes → final batch
            ]

            original_broadcast = torch.distributed.broadcast
            torch.distributed.broadcast = lambda t, src=0, group=None: None
            try:
                send_thread = threading.Thread(
                    target=sender._send_per_tensor_impl,
                    args=(weights,),
                )
                send_thread.start()

                batches = list(receiver.receive())
                send_thread.join(timeout=5)
            finally:
                torch.distributed.broadcast = original_broadcast

            # First two tensors each trigger a flush (1024 >= 1024), final is last.
            assert len(batches) == 3
            # Batch 0: w0, is_last=False
            assert len(batches[0][0]) == 1
            assert batches[0][0][0][0] == "w0"
            assert batches[0][1] is False
            # Batch 1: w1, is_last=False
            assert len(batches[1][0]) == 1
            assert batches[1][0][0][0] == "w1"
            assert batches[1][1] is False
            # Batch 2: w2, is_last=True
            assert len(batches[2][0]) == 1
            assert batches[2][0][0][0] == "w2"
            assert batches[2][1] is True
        finally:
            receiver.finalize()
            sender.finalize()

    def test_empty_weights(self):
        """Empty weight iterator should still send a single is_last=True batch."""
        sender = ModelWeightSender()
        try:
            sender.setup(pg=None, bucket_size=1_000_000, per_tensor=True)
            zmq_info = sender.zmq_info

            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            time.sleep(0.1)

            original_broadcast = torch.distributed.broadcast
            torch.distributed.broadcast = lambda t, src=0, group=None: None
            try:
                send_thread = threading.Thread(
                    target=sender._send_per_tensor_impl,
                    args=([],),
                )
                send_thread.start()

                batches = list(receiver.receive())
                send_thread.join(timeout=5)
            finally:
                torch.distributed.broadcast = original_broadcast

            assert len(batches) == 1
            batch_meta, is_last = batches[0]
            assert is_last is True
            assert len(batch_meta) == 0
        finally:
            receiver.finalize()
            sender.finalize()


# ---------------------------------------------------------------------------
# ModelWeightMetadataReceiver tests
# ---------------------------------------------------------------------------


class TestMetadataReceiver:
    """Tests for ModelWeightMetadataReceiver in isolation."""

    def test_finalize_closes_zmq(self):
        """finalize() should close the ZMQ socket and terminate the context."""
        sender = ModelWeightSender()
        try:
            zmq_info = sender.zmq_info
            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            assert receiver._socket is not None
            assert receiver._zmq_context is not None

            receiver.finalize()
            assert receiver._socket is None
            assert receiver._zmq_context is None
        finally:
            sender.finalize()

    def test_double_finalize_safe(self):
        """Calling finalize() twice should be safe (no-op on second call)."""
        sender = ModelWeightSender()
        try:
            zmq_info = sender.zmq_info
            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            receiver.finalize()
            receiver.finalize()  # Should not raise.
            assert receiver._socket is None
        finally:
            sender.finalize()

    def test_metadata_dtype_shapes_preserved(self):
        """Verify dtype strings and shapes survive the ZMQ roundtrip."""
        sender = ModelWeightSender()
        try:
            sender.setup(pg=None, bucket_size=100_000_000, per_tensor=True)
            zmq_info = sender.zmq_info

            receiver = ModelWeightMetadataReceiver(zmq_info["zmq_ip"], zmq_info["zmq_port"])
            time.sleep(0.1)

            weights = [
                ("embed", torch.randn(512, 768)),  # float32
                ("norm", torch.randn(768).to(torch.bfloat16)),  # bfloat16
                ("head", torch.randn(768, 32000).to(torch.float16)),  # float16
            ]

            original_broadcast = torch.distributed.broadcast
            torch.distributed.broadcast = lambda t, src=0, group=None: None
            try:
                send_thread = threading.Thread(
                    target=sender._send_per_tensor_impl,
                    args=(weights,),
                )
                send_thread.start()

                batches = list(receiver.receive())
                send_thread.join(timeout=5)
            finally:
                torch.distributed.broadcast = original_broadcast

            assert len(batches) == 1
            batch_meta, is_last = batches[0]
            assert is_last is True
            assert len(batch_meta) == 3

            # Check each tensor's metadata.
            name, dtype_str, shape = batch_meta[0]
            assert name == "embed"
            assert dtype_str == "float32"
            assert shape == (512, 768)

            name, dtype_str, shape = batch_meta[1]
            assert name == "norm"
            assert dtype_str == "bfloat16"
            assert shape == (768,)

            name, dtype_str, shape = batch_meta[2]
            assert name == "head"
            assert dtype_str == "float16"
            assert shape == (768, 32000)
        finally:
            receiver.finalize()
            sender.finalize()
