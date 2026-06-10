# -*- coding: utf-8 -*-
"""Streaming safetensors writer — saves one tensor at a time to avoid OOM.

Standard ``safetensors.torch.save_file`` requires the entire state dict in
memory.  For 70B+ models this can exceed available CPU RAM.  This module
provides :func:`save_safetensors_streaming` which iterates a tensor generator
and writes each tensor directly to disk, keeping only one tensor in memory at
any point.

The implementation uses a **seek-back** approach:

1. Reserve space for the safetensors header (generously estimated).
2. Write tensor data sequentially, collecting metadata along the way.
3. Build the JSON header and seek back to the beginning to write it.

The resulting file is a fully valid safetensors file that can be loaded by
``safetensors.torch.load_file`` (which uses mmap internally).

Writes go through the OS page cache, so the function returns quickly after
the last tensor is written.  Callers can hand off ``fsync`` + ``rename`` to
a background thread for minimal main-thread blocking.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Iterable

import torch

# ---------------------------------------------------------------------------
# Dtype mapping (PyTorch → safetensors string / element size)
# ---------------------------------------------------------------------------

DTYPE_TO_SAFETENSORS: dict[torch.dtype, str] = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

DTYPE_ELEMENT_SIZE: dict[str, int] = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a CPU tensor to its raw bytes in C-contiguous order."""
    t = tensor.detach().contiguous()
    # numpy() handles the dtype↔raw mapping correctly for all types,
    # including bfloat16 (via numpy ≥ 1.24 or torch's internal conversion).
    try:
        return t.numpy().tobytes()
    except (RuntimeError, TypeError):
        # Fallback for dtypes not natively supported by numpy (e.g. bfloat16
        # on older numpy).  View as uint8 to get the raw byte representation.
        return t.view(torch.uint8).numpy().tobytes()


def save_safetensors_streaming(
    tensor_iter: Iterable[tuple[str, torch.Tensor]],
    filepath: str | os.PathLike,
    *,
    estimated_tensor_count: int = 0,
    rename: bool = True,
) -> str:
    """Stream-write a safetensors file, holding only one tensor in memory.

    Args:
        tensor_iter: Yields ``(name, tensor)`` pairs.  Each tensor must
            already reside on CPU.  After the tensor's bytes are written it
            is no longer referenced — the caller should ``del`` the tensor
            from their side as well.
        filepath: Destination path.  Data is first written to
            ``filepath + ".tmp"``; if *rename* is ``True`` the temp file is
            atomically renamed to *filepath* before returning.
        estimated_tensor_count: Approximate number of tensors (for header
            space reservation).  ``0`` uses a generous default.
            Overestimating is harmless (extra whitespace padding in the
            header); underestimating triggers a transparent rewrite.
        rename: If ``True`` (default), atomically replace *filepath* with
            the completed temp file.  If ``False``, leave the temp file on
            disk — the caller is responsible for ``fsync`` + ``rename``
            (useful for hybrid async mode).

    Returns:
        The path actually written: *filepath* when ``rename=True``,
        otherwise ``str(filepath) + ".tmp"``.
    """
    filepath = str(filepath)
    tmp_path = filepath + ".tmp"

    # --- Header space estimation ---
    # Each tensor entry in the JSON header looks like:
    #   "name": {"dtype": "BF16", "shape": [4096, 4096], "data_offsets": [0, 33554432]}
    # Typical entry is ~80-150 bytes.  200 per entry is generous.
    count_hint = max(estimated_tensor_count, 256)
    max_header_size = count_hint * 200 + 256
    # Align to 8 bytes (safetensors convention).
    max_header_size = ((max_header_size + 7) // 8) * 8

    header: dict[str, dict] = {}
    data_offset = 0

    with open(tmp_path, "w+b") as f:
        # Phase 1: Reserve space for header_size (8B) + header JSON.
        f.write(b"\x00" * (8 + max_header_size))

        # Phase 2: Stream tensor data.
        for name, tensor in tensor_iter:
            dtype_str = DTYPE_TO_SAFETENSORS.get(tensor.dtype)
            if dtype_str is None:
                raise ValueError(
                    f"Unsupported dtype {tensor.dtype} for tensor '{name}'. "
                    f"Supported: {list(DTYPE_TO_SAFETENSORS.keys())}"
                )

            raw = _tensor_to_bytes(tensor)
            nbytes = len(raw)

            header[name] = {
                "dtype": dtype_str,
                "shape": list(tensor.shape),
                "data_offsets": [data_offset, data_offset + nbytes],
            }

            f.write(raw)
            data_offset += nbytes
            # Explicitly delete to help GC release memory promptly.
            del raw, tensor

        # Phase 3: Build header JSON and write it back at the start.
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

        if len(header_json) <= max_header_size:
            # Common case: header fits in the reserved space.
            # Pad with spaces to fill exactly max_header_size bytes.
            padded = header_json + b" " * (max_header_size - len(header_json))
            f.seek(0)
            f.write(struct.pack("<Q", max_header_size))
            f.write(padded)
        else:
            # Rare case: header overflows the reserved space.
            # Must rewrite the entire file with a larger header.
            _rewrite_with_larger_header(f, header_json, data_offset, tmp_path)

    if rename:
        os.replace(tmp_path, filepath)
        return filepath
    return tmp_path


def _rewrite_with_larger_header(
    original_f,
    header_json: bytes,
    data_size: int,
    tmp_path: str,
) -> None:
    """Rewrite the file when the header exceeds the reserved space.

    This is a fallback for the rare case where the actual header is larger
    than the estimated reservation.  We write to a second temp file and
    then replace the original.
    """
    actual_header_size = len(header_json)
    # Pad to 8-byte alignment.
    actual_header_size = ((actual_header_size + 7) // 8) * 8
    padded = header_json + b" " * (actual_header_size - len(header_json))

    rewrite_path = tmp_path + ".rewrite"
    try:
        with open(rewrite_path, "wb") as out:
            out.write(struct.pack("<Q", actual_header_size))
            out.write(padded)

            # Copy tensor data from the original file.
            # The original file has: 8B header_size + old_reserved + tensor_data
            # We need to find where tensor data starts.
            original_f.seek(0, 2)  # seek to end
            file_size = original_f.tell()
            tensor_data_start = file_size - data_size
            original_f.seek(tensor_data_start)

            # Copy in chunks to avoid large memory allocations.
            chunk_size = 64 * 1024 * 1024  # 64 MB
            remaining = data_size
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = original_f.read(to_read)
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)

        # Replace original temp file with the rewritten one.
        os.replace(rewrite_path, tmp_path)
    except Exception:
        # Clean up on failure.
        if os.path.exists(rewrite_path):
            os.unlink(rewrite_path)
        raise
