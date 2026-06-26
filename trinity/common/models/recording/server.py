"""Shared HTTP server wiring for generation recording."""

import logging

from trinity.common.models.recording.context import RecordingIdentityMiddleware
from trinity.common.models.recording.query import (
    RECORDER_STATE_ATTR,
    STORE_STATE_ATTR,
    query_router,
)
from trinity.common.models.recording.recorder import Recorder


def add_recording_middleware(app) -> None:
    """Install recording middleware before serving.

    Some FastAPI/Starlette integrations build ``middleware_stack`` before
    uvicorn starts serving. Clearing the cached stack lets Starlette rebuild it
    with our middleware on first request.
    """
    if getattr(app, "middleware_stack", None) is not None:
        app.middleware_stack = None
    app.add_middleware(RecordingIdentityMiddleware)


def mount_recording_api(
    app,
    recorder: Recorder,
    logger: logging.Logger,
    *,
    engine_name: str,
    start_recorder: bool = False,
) -> None:
    """Mount recording middleware/query routes and expose state to handlers."""
    add_recording_middleware(app)
    app.include_router(query_router)

    setattr(app.state, STORE_STATE_ATTR, recorder.store)
    setattr(app.state, RECORDER_STATE_ATTR, recorder)

    if start_recorder:
        recorder.start()

    logger.info(
        "%s generation recording enabled: store=%s rank=%d",
        engine_name,
        type(recorder.store).__name__,
        recorder.rank,
    )
