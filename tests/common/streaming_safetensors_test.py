# -*- coding: utf-8 -*-
"""Tests for the streaming safetensors writer."""

import json
import os
import struct
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from trinity.common.models.streaming_safetensors import save_safetensors_streaming


class TestSaveSafetensorsStreaming(unittest.TestCase):
    """Unit tests for :func:`save_safetensors_streaming`."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_basic_roundtrip(self):
        """Write several tensors and verify they load back identically."""
        tensors = {
            "layer.0.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer.0.bias": torch.randn(64, dtype=torch.float32),
            "layer.1.weight": torch.randn(32, 64, dtype=torch.float32),
        }
        filepath = os.path.join(self.tmpdir, "model.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_bfloat16_roundtrip(self):
        """bf16 tensors survive the streaming write."""
        tensors = {
            "w1": torch.randn(128, 128, dtype=torch.bfloat16),
            "w2": torch.randn(128, dtype=torch.bfloat16),
        }
        filepath = os.path.join(self.tmpdir, "bf16.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        loaded = load_file(filepath)
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_mixed_dtypes(self):
        """Different dtypes in the same file."""
        tensors = {
            "fp32": torch.randn(16, dtype=torch.float32),
            "fp16": torch.randn(16, dtype=torch.float16),
            "bf16": torch.randn(16, dtype=torch.bfloat16),
            "int8": torch.randint(-128, 127, (16,), dtype=torch.int8),
            "int32": torch.randint(0, 100, (16,), dtype=torch.int32),
        }
        filepath = os.path.join(self.tmpdir, "mixed.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    def test_empty_tensor(self):
        """A tensor with zero elements should still round-trip."""
        tensors = {
            "empty": torch.randn(0, 64, dtype=torch.float32),
            "nonempty": torch.randn(4, 64, dtype=torch.float32),
        }
        filepath = os.path.join(self.tmpdir, "empty.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        loaded = load_file(filepath)
        self.assertEqual(loaded["empty"].shape, (0, 64))
        torch.testing.assert_close(loaded["nonempty"], tensors["nonempty"])

    def test_single_tensor(self):
        """Edge case: a file with exactly one tensor."""
        tensors = {"only": torch.randn(1024, dtype=torch.float32)}
        filepath = os.path.join(self.tmpdir, "single.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        loaded = load_file(filepath)
        torch.testing.assert_close(loaded["only"], tensors["only"])

    # ------------------------------------------------------------------
    # Generator semantics
    # ------------------------------------------------------------------

    def test_generator_input(self):
        """Accepts a lazy generator, not just a dict.items()."""
        filepath = os.path.join(self.tmpdir, "gen.safetensors")

        def gen():
            for i in range(5):
                yield f"param_{i}", torch.randn(32, 32, dtype=torch.bfloat16)

        # We need to capture the generated tensors for comparison.
        reference = {}

        def gen_with_capture():
            for i in range(5):
                t = torch.randn(32, 32, dtype=torch.bfloat16)
                reference[f"param_{i}"] = t.clone()
                yield f"param_{i}", t

        save_safetensors_streaming(gen_with_capture(), filepath)

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(reference.keys()))
        for name in reference:
            torch.testing.assert_close(loaded[name], reference[name])

    # ------------------------------------------------------------------
    # rename=False (hybrid async mode)
    # ------------------------------------------------------------------

    def test_rename_false_leaves_tmp(self):
        """With rename=False, the .tmp file is left for the caller."""
        filepath = os.path.join(self.tmpdir, "model.safetensors")
        tensors = {"w": torch.randn(8, 8)}

        result = save_safetensors_streaming(tensors.items(), filepath, rename=False)

        self.assertEqual(result, filepath + ".tmp")
        self.assertTrue(os.path.exists(filepath + ".tmp"))
        self.assertFalse(os.path.exists(filepath))

        # The .tmp file should be a valid safetensors file.
        loaded = load_file(filepath + ".tmp")
        torch.testing.assert_close(loaded["w"], tensors["w"])

    # ------------------------------------------------------------------
    # Header overflow fallback
    # ------------------------------------------------------------------

    def test_header_overflow_fallback(self):
        """When estimated_tensor_count is too low, the rewrite fallback kicks in."""
        # Use a very small estimate to force overflow.
        tensors = {f"very_long_tensor_name_{i:04d}": torch.randn(4) for i in range(200)}
        filepath = os.path.join(self.tmpdir, "overflow.safetensors")

        save_safetensors_streaming(
            tensors.items(),
            filepath,
            estimated_tensor_count=1,  # way too small
        )

        loaded = load_file(filepath)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, expected in tensors.items():
            torch.testing.assert_close(loaded[name], expected)

    # ------------------------------------------------------------------
    # File format validation
    # ------------------------------------------------------------------

    def test_valid_safetensors_header(self):
        """The header contains valid JSON with correct dtype/shape/offsets."""
        tensors = {
            "a": torch.randn(4, 8, dtype=torch.float16),
            "b": torch.randn(16, dtype=torch.bfloat16),
        }
        filepath = os.path.join(self.tmpdir, "header.safetensors")

        save_safetensors_streaming(tensors.items(), filepath)

        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size)

        header = json.loads(header_json)
        self.assertIn("a", header)
        self.assertIn("b", header)

        self.assertEqual(header["a"]["dtype"], "F16")
        self.assertEqual(header["a"]["shape"], [4, 8])

        self.assertEqual(header["b"]["dtype"], "BF16")
        self.assertEqual(header["b"]["shape"], [16])

        # Offsets should be contiguous.
        self.assertEqual(header["a"]["data_offsets"][0], 0)
        a_end = header["a"]["data_offsets"][1]
        self.assertEqual(a_end, 4 * 8 * 2)  # 4*8 elements * 2 bytes (F16)
        self.assertEqual(header["b"]["data_offsets"][0], a_end)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_unsupported_dtype_raises(self):
        """Unsupported dtypes should raise a clear error."""

        def gen():
            yield "bad", torch.randn(4, dtype=torch.complex64)

        filepath = os.path.join(self.tmpdir, "bad.safetensors")
        with self.assertRaises(ValueError):
            save_safetensors_streaming(gen(), filepath)


if __name__ == "__main__":
    unittest.main()
