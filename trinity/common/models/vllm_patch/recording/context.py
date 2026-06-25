"""Per-request recording identity propagation.

The OpenAI ``Authorization: Bearer <api_key>`` header is the recording identity
source because it works with CLI agents that do not support custom headers. We
read the identity in an in-process ASGI middleware and stash it in a contextvar
so the engine-level wrapper (which runs in the same async task as the serving
handler) can recover it at record time.

The identity value is the **record key** — the MemoryStore group key under
which a whole reward unit's worth of turns (one sample / one episode) is
recorded, and the key the runner later reports ``{record_key: reward}`` against
so the coordinator can join the reward inside the store.

No identity header on a request is fine: the recorder falls back to
``request_id`` as the record key so nothing is silently dropped.
"""
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# The record key for the in-flight request (the MemoryStore group key), or None
# when the client did not send a supported identity header (the recorder then
# uses request_id as a fallback record key).
record_key_ctx: ContextVar[Optional[str]] = ContextVar("trinity_recording_record_key", default=None)

# Set around auxiliary engine.generate calls (logprobs recomputation,
# convert_messages_to_experience) so the recorder skips them — those 1-token
# forwards are not real turns and would pollute the store.
skip_recording_ctx: ContextVar[bool] = ContextVar("trinity_recording_skip", default=False)

#: Preferred identity header for OpenAI-compatible clients.
AUTHORIZATION_HEADER = "authorization"


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


def get_recording_record_key(request: Request) -> Optional[str]:
    """Return the recording identity for a request.

    OpenAI-compatible API keys are used because every supported agent platform
    can pass them.
    """
    return extract_bearer_token(request.headers.get(AUTHORIZATION_HEADER))


class RecordingIdentityMiddleware(BaseHTTPMiddleware):
    """Capture request identity into ``record_key_ctx`` for the request's lifetime.

    Runs in-process (ASGI) — no extra network hop, no serialization cost beyond
    a contextvar set/reset.
    """

    async def dispatch(self, request: Request, call_next):
        record_key = get_recording_record_key(request)
        token = record_key_ctx.set(record_key)
        try:
            return await call_next(request)
        finally:
            record_key_ctx.reset(token)
