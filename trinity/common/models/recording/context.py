"""Per-request recording identity propagation shared by model engines."""

from contextvars import ContextVar
from typing import Any, Optional

try:
    from starlette.middleware.base import BaseHTTPMiddleware
except ModuleNotFoundError:
    BaseHTTPMiddleware = object  # type: ignore

# The record key for the in-flight request (the MemoryStore group key), or None
# when the client did not send a supported identity header.
record_key_ctx: ContextVar[Optional[str]] = ContextVar("trinity_recording_record_key", default=None)

# Set around auxiliary generate calls (logprobs recomputation, message
# conversion) so recorders skip them.
skip_recording_ctx: ContextVar[bool] = ContextVar("trinity_recording_skip", default=False)

AUTHORIZATION_HEADER = "authorization"


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an Authorization header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def get_recording_record_key(request: Any) -> Optional[str]:
    """Return the recording identity for an HTTP request."""
    return extract_bearer_token(request.headers.get(AUTHORIZATION_HEADER))


class RecordingIdentityMiddleware(BaseHTTPMiddleware):
    """Capture request identity into ``record_key_ctx`` for the request lifetime."""

    async def dispatch(self, request: Any, call_next):
        record_key = get_recording_record_key(request)
        token = record_key_ctx.set(record_key)
        try:
            return await call_next(request)
        finally:
            record_key_ctx.reset(token)
