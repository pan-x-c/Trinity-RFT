"""Per-request recording identity propagation.

The OpenAI ``Authorization: Bearer <api_key>`` header is the preferred identity
source for recording because it works with CLI agents that do not support
custom headers. ``X-Session-ID`` remains a compatibility fallback. We read the
identity in an in-process ASGI middleware and stash it in a contextvar so the
engine-level wrapper (which runs in the same async task as the serving handler)
can recover it at record time.

No identity header on a request is fine: the recorder falls back to
``request_id`` as the task id so nothing is silently dropped.
"""
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# The task id for the in-flight request, or None when the client did not send
# a supported identity header (the recorder then uses request_id as a fallback
# task id).
task_id_ctx: ContextVar[Optional[str]] = ContextVar("trinity_recording_task_id", default=None)

#: Preferred identity header for OpenAI-compatible clients.
AUTHORIZATION_HEADER = "authorization"
#: Compatibility identity header. Lower-cased per ASGI/httpx convention.
SESSION_ID_HEADER = "x-session-id"


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an Authorization header.

    Returns None when the header is missing or does not use the Bearer scheme.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def get_recording_task_id(request: Request) -> Optional[str]:
    """Return the recording identity for a request.

    Prefer OpenAI-compatible API keys because every supported agent platform
    can pass them. ``X-Session-ID`` is kept as a backward-compatible fallback
    for direct clients and debugging.
    """
    return extract_bearer_token(request.headers.get(AUTHORIZATION_HEADER)) or request.headers.get(
        SESSION_ID_HEADER
    )


class RecordingIdentityMiddleware(BaseHTTPMiddleware):
    """Capture request identity into ``task_id_ctx`` for the request's lifetime.

    Runs in-process (ASGI) — no extra network hop, no serialization cost beyond
    a contextvar set/reset.
    """

    async def dispatch(self, request: Request, call_next):
        task_id = get_recording_task_id(request)
        token = task_id_ctx.set(task_id)
        try:
            return await call_next(request)
        finally:
            task_id_ctx.reset(token)


# Backward-compatible export for existing imports.
SessionMiddleware = RecordingIdentityMiddleware
