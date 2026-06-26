"""Wiring that installs SGLang generation recording onto the embedded HTTP server.

Mirrors ``trinity/common/models/vllm_patch/recording/server.py:_setup_recording``:
(1) the engine wrap (``create_sglang_recorder``), (2) ``RecordingIdentityMiddleware``
— an in-process ASGI middleware reading ``Authorization: Bearer <record_key>``
into a contextvar, and (3) ``query_router`` — ``/records/*`` endpoints.

Called from ``sglang_patch.server_patch.get_api_server`` after the
``tokenizer_manager`` is created and **before** the uvicorn task starts serving,
so the middleware/router are mounted on ``app`` in time. The recorder and store
are owned by ``SGLangRolloutModel`` (passed in) so it can drain them in-process
via ``extract_experience_from_history``; they are also stashed on ``app.state``
for the ``query_router`` HTTP drain path used by the coordinator.
"""
import logging
from typing import Optional, Tuple

from trinity.common.models.recording.context import RecordingIdentityMiddleware
from trinity.common.models.recording.query import (
    RECORDER_STATE_ATTR,
    STORE_STATE_ATTR,
    query_router,
)
from trinity.common.models.recording.recorder import Recorder
from trinity.common.models.recording.store import RecordStore
from trinity.common.models.sglang_patch.recording.recorder import create_sglang_recorder


def _add_recording_middleware(app) -> None:
    """Install recording middleware before serving, even if SGLang built the stack.

    Starlette rejects ``add_middleware`` after ``middleware_stack`` is built with
    "Cannot add middleware after an application has started". Clearing the cached
    stack lets Starlette rebuild it with our middleware on first request (same
    defensive pattern as the vLLM recording patch).
    """
    if getattr(app, "middleware_stack", None) is not None:
        app.middleware_stack = None
    app.add_middleware(RecordingIdentityMiddleware)


def setup_sglang_recording(
    tokenizer_manager,
    app,
    logger: logging.Logger,
    *,
    recorder: Optional[Recorder] = None,
    store: Optional[RecordStore] = None,
    routed_experts_layout: Optional[Tuple[int, int]] = None,
) -> Recorder:
    """Wire generation recording onto the in-construction SGLang server.

    Only called when recording is on. The recorder is started here (its flusher
    task lives in the server's event loop, same loop as ``SGLangRolloutModel``).
    """
    recorder = create_sglang_recorder(
        tokenizer_manager,
        logger,
        store=store,
        recorder=recorder,
        enabled=True,
        routed_experts_layout=routed_experts_layout,
    )

    # (2) in-process middleware: Authorization bearer -> record_key contextvar.
    _add_recording_middleware(app)

    # (3) query routes mounted on the main app; OpenAI /v1/* surface untouched.
    app.include_router(query_router)

    setattr(app.state, STORE_STATE_ATTR, recorder.store)
    setattr(app.state, RECORDER_STATE_ATTR, recorder)

    recorder.start()

    logger.info(
        "SGLang generation recording enabled: store=%s",
        type(recorder.store).__name__,
    )
    return recorder
