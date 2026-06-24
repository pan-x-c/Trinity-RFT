"""Per-request task_id propagation.

The OpenAI ``X-Session-ID`` request header carries the task id that scopes a
multi-turn task. We read it in an in-process ASGI middleware and stash it in a
contextvar so the engine-level wrapper (which runs in the same async task as
the serving handler) can recover it at record time.

No ``X-Session-ID`` on a request is fine: the recorder falls back to
``request_id`` as the task id so nothing is silently dropped.
"""
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# The task id for the in-flight request, or None when the client did not send
# ``X-Session-ID`` (the recorder then uses request_id as a fallback task id).
task_id_ctx: ContextVar[Optional[str]] = ContextVar("trinity_recording_task_id", default=None)

#: Canonical header name. Lower-cased per ASGI/httpx convention.
SESSION_ID_HEADER = "x-session-id"


class SessionMiddleware(BaseHTTPMiddleware):
    """Capture ``X-Session-ID`` into ``task_id_ctx`` for the request's lifetime.

    Runs in-process (ASGI) — no extra network hop, no serialization cost beyond
    a contextvar set/reset.
    """

    async def dispatch(self, request: Request, call_next):
        task_id = request.headers.get(SESSION_ID_HEADER)
        token = task_id_ctx.set(task_id)
        try:
            return await call_next(request)
        finally:
            task_id_ctx.reset(token)
