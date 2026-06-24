"""Generation-recording patch for the vLLM OpenAI server.

Self-contained entry point that mirrors ``api_patch_v17.py``'s bootstrap flow
(``build_app`` -> ``init_app_state`` -> ``serve_http``) and additionally wires
in generation recording without touching vLLM source code.

Designed for vllm >= 0.17.0. Drop-in alternative to
``trinity.common.models.vllm_patch.api_patch_v17``: point your launcher at
``recording.run_api_server_with_recording`` instead.
"""
from trinity.common.models.vllm_patch.recording.config import RecordingConfig
from trinity.common.models.vllm_patch.recording.context import (
    SessionMiddleware,
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
    "SqlStore",
    "Recorder",
    "SessionMiddleware",
    "build_experience",
    "patch_engine_for_recording",
    "query_router",
    "run_api_server_with_recording",
    "task_id_ctx",
]
