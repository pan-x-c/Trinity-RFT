# -*- coding: utf-8 -*-
"""Core data structures and utilities for bucketed weight transfer.

Provides TensorMeta (bucket metadata for each tensor chunk) and async generator
utilities for splitting large tensors into fixed-size bucket chunks and
reassembling them on the receiver side.

Ported from veRL's checkpoint_engine/base.py with all external dependencies
removed — only requires torch.
"""
from dataclasses import dataclass
from typing import AsyncGenerator, Generator, Iterable, Tuple, Union

import torch


@dataclass
class TensorMeta:
    """Metadata for a single tensor chunk within a bucket.

    When a tensor is too large to fit in one bucket, it is split into multiple
    chunks. Each chunk carries a TensorMeta describing where it came from
    (name, shape, dtype, chunk_offset within the original tensor) and where it
    sits in the bucket buffer (offset, chunk_size).
    """

    name: str
    """The parameter name of the weight tensor."""
    shape: torch.Size
    """The shape of the full (unsplit) weight tensor."""
    dtype: torch.dtype
    """The dtype of the weight tensor."""
    chunk_offset: int
    """Byte offset of this chunk within the original tensor's flat byte view."""
    chunk_size: int
    """Size of this chunk in bytes."""
    offset: int | None
    """Byte offset of this chunk within the bucket buffer. Set by the sender."""


async def _ensure_async_iterator(iterable):
    """Convert a sync or async iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


async def split_weight_chunks(
    weights: Union[
        Generator[Tuple[str, torch.Tensor], None, None],
        Iterable[Tuple[str, torch.Tensor]],
    ],
    bucket_size: int,
) -> AsyncGenerator[Tuple[TensorMeta, torch.Tensor], None]:
    """Split weight tensors into bucket-sized byte chunks.

    Each weight tensor is viewed as a flat uint8 buffer and sliced into
    chunks of at most ``bucket_size`` bytes.  Large tensors that exceed the
    bucket size are split across multiple chunks (tracked via
    ``chunk_offset``).

    Args:
        weights: An iterable (sync or async) yielding ``(name, tensor)``
            pairs — e.g. from ``engine.get_per_tensor_param()``.
        bucket_size: Maximum bucket size in bytes.

    Yields:
        ``(TensorMeta, chunk_view)`` where *chunk_view* is a uint8 tensor
        slice of the flattened weight.
    """
    async for name, weight in _ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=None,
            )
            yield (tensor_meta, buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size


async def merge_weight_chunks(
    chunks: AsyncGenerator[Tuple[TensorMeta, torch.Tensor], None],
    bucket_size: int,
) -> AsyncGenerator[Tuple[str, torch.Tensor], None]:
    """Reassemble weight tensors from bucket chunks.

    Inverse of :func:`split_weight_chunks`.  Small tensors (fitting in a
    single bucket) are returned directly via a dtype + shape view.  Large
    tensors spanning multiple chunks are accumulated into a pre-allocated
    buffer and yielded once all chunks arrive.

    Args:
        chunks: An async generator yielding ``(TensorMeta, chunk_bytes)``
            pairs, where *chunk_bytes* is a uint8 tensor.
        bucket_size: The bucket size used during splitting.

    Yields:
        ``(name, tensor)`` with the original dtype and shape.
    """
    merge_name: str | None = None
    merge_weight: torch.Tensor | None = None
    merge_buffer: torch.Tensor | None = None
    merge_offset: int = 0

    async for tensor_meta, chunk in _ensure_async_iterator(chunks):
        assert chunk.dtype == torch.uint8, f"Chunk dtype must be uint8, but got {chunk.dtype}"
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize

        # Weight fits in one bucket — zero-copy view.
        if nbytes <= bucket_size:
            assert merge_weight is None, f"Previous large tensor {merge_name!r} not fully merged"
            name = tensor_meta.name
            weight = chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            yield (name, weight)
            continue

        # Large tensor spanning multiple buckets — accumulate.
        if merge_weight is None:
            assert tensor_meta.chunk_offset == 0, f"First chunk offset must be 0, got {tensor_meta}"
            merge_name = tensor_meta.name
            merge_weight = torch.empty(
                tensor_meta.shape,
                dtype=tensor_meta.dtype,
                device=chunk.device,
            )
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0

        assert tensor_meta.name == merge_name
        assert merge_offset == tensor_meta.chunk_offset
        merge_buffer[  # type: ignore[index]
            tensor_meta.chunk_offset : tensor_meta.chunk_offset + tensor_meta.chunk_size
        ] = chunk
        merge_offset += tensor_meta.chunk_size

        if tensor_meta.chunk_offset + tensor_meta.chunk_size == nbytes:
            yield (merge_name, merge_weight)  # type: ignore[arg-type]
            merge_name, merge_weight, merge_buffer, merge_offset = (
                None,
                None,
                None,
                0,
            )
