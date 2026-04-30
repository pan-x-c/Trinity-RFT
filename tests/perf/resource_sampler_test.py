"""Tests for perf resource timeline helpers."""

import itertools
import time
import unittest

from trinity.perf.report_utils import build_resource_timeline_payload
from trinity.perf.resource_backends import GPUSample, ResourceSample
from trinity.perf.resource_sampler import ResourceSampler


class FakeBackend:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.sample_index = itertools.count()

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def sample(self):
        index = next(self.sample_index)
        return ResourceSample(
            timestamp=1000.0 + index,
            cpu_percent=50.0 + index,
            memory_rss_mb=1024.0 + index,
            memory_percent=20.0 + index,
            gpu_metrics=[
                GPUSample(
                    gpu_id=0,
                    name="GPU-0",
                    gpu_util_percent=70.0 + index,
                    gpu_memory_used_mb=16000.0 + index,
                    gpu_memory_total_mb=24000.0,
                ),
                GPUSample(
                    gpu_id=1,
                    name="GPU-1",
                    gpu_util_percent=75.0 + index,
                    gpu_memory_used_mb=15000.0 + index,
                    gpu_memory_total_mb=24000.0,
                ),
            ],
        )


class ResourceSamplerTest(unittest.TestCase):
    def test_resource_sampler_collects_samples(self):
        backend = FakeBackend()
        sampler = ResourceSampler(interval_seconds=0.01, backend=backend)

        sampler.start()
        time.sleep(0.03)
        samples = sampler.stop()

        self.assertTrue(backend.opened)
        self.assertTrue(backend.closed)
        self.assertGreaterEqual(len(samples), 2)
        self.assertEqual(samples[0].gpu_metrics[0].gpu_id, 0)

    def test_build_resource_timeline_payload_keeps_cpu_single_line_and_gpu_per_device(self):
        samples = [FakeBackend().sample(), FakeBackend().sample()]

        payload = build_resource_timeline_payload(samples)

        self.assertEqual(len(payload["resource_timeline"]), 2)
        self.assertEqual(len(payload["chart_series"]["cpu_percent"]), 2)
        self.assertEqual(set(payload["chart_series"]["gpu_util_percent"].keys()), {"0", "1"})
        self.assertEqual(
            payload["chart_series"]["gpu_memory_used_mb"]["0"]["name"],
            "GPU-0",
        )
