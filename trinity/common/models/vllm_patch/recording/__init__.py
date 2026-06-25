"""Generation-recording patch for the vLLM OpenAI server.
Designed for vllm >= 0.23.0.
"""
from trinity.common.models.vllm_patch.recording.config import RecordingConfig
from trinity.common.models.vllm_patch.recording.context import (
    RecordingIdentityMiddleware,
    task_id_ctx,
)
from trinity.common.models.vllm_patch.recording.models import build_experience
from trinity.common.models.vllm_patch.recording.query import query_router
from trinity.common.models.vllm_patch.recording.recorder import (
    Recorder,
    patch_engine_for_recording,
)
from trinity.common.models.vllm_patch.recording.server import (
    run_api_server_with_recording,
)
from trinity.common.models.vllm_patch.recording.store import (
    MemoryStore,
    RecordStore,
    SqlStore,
)

__all__ = [
    "MemoryStore",
    "RecordStore",
    "RecordingConfig",
    "RecordingIdentityMiddleware",
    "SqlStore",
    "Recorder",
    "build_experience",
    "patch_engine_for_recording",
    "query_router",
    "run_api_server_with_recording",
    "task_id_ctx",
]
