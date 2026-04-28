"""Performance tooling package for Trinity."""

from .report_utils import build_resource_timeline_payload
from .resource_sampler import ResourceSampler
from .stage_perf import (
    ExplorerPerfOptions,
    run_explorer_perf,
    write_explorer_perf_output,
)
from .tensorboard_metrics import (
    TensorBoardScalarReader,
    collect_step_metrics,
)

__all__ = [
    "ExplorerPerfOptions",
    "ResourceSampler",
    "TensorBoardScalarReader",
    "build_resource_timeline_payload",
    "collect_step_metrics",
    "run_explorer_perf",
    "write_explorer_perf_output",
]
