"""HTTP query endpoints over recorded experiences.

Mounted on the main vLLM app (we own ``app`` in the bootstrap) via
``app.include_router(query_router)``. Routes are prefixed under ``/records`` to
avoid colliding with the OpenAI ``/v1/*`` surface — the OpenAI protocol is left
untouched.

Listing endpoints return ``Experience.to_dict()`` (lightweight metadata, no
tensor payload). A single-turn endpoint returns the full experience as Trinity
serialized bytes (``Experience.serialize()``) so it can be fed straight back
into Trinity pipelines.
"""
from fastapi import APIRouter, HTTPException, Request, Response

from trinity.common.experience import Experience
from trinity.common.models.vllm_patch.recording.store import RecordStore

#: Mounted on app.state by the bootstrap wiring in ``server.py``.
_STORE_ATTR = "trinity_record_store"

query_router = APIRouter(prefix="/records", tags=["trinity-recording"])


def _store(request: Request, *, read: bool = False) -> RecordStore:
    store = getattr(request.app.state, _STORE_ATTR, None)
    if store is None:
        raise HTTPException(status_code=503, detail="recording store not configured")
    if read and not getattr(store, "supports_reads", True):
        # SqlStore shares the explorer proxy's table; reads/consumes are served
        # by the proxy's /feedback + /commit, not by these debug endpoints.
        raise HTTPException(
            status_code=503,
            detail="this record store backend is read-only from the vLLM "
            "side; query via the explorer proxy instead",
        )
    return store


async def _get_exp(store: RecordStore, task_id: str, request_id: str) -> Experience:
    exp = await store.get_turn(task_id, request_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experience not found")
    return exp


@query_router.get("/tasks")
async def list_tasks(request: Request) -> dict:
    """List all known task ids."""
    store = _store(request, read=True)
    return {"task_ids": await store.list_tasks()}


@query_router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    """Return lightweight metadata for all experiences of a task.

    Tensor payloads (tokens/logprobs/routed_experts) are omitted here to keep
    listing responses small; fetch the per-turn blob endpoint for full data.
    """
    store = _store(request, read=True)
    experiences = await store.get_task(task_id)
    return {"task_id": task_id, "turns": [e.to_dict() for e in experiences]}


@query_router.get("/tasks/{task_id}/turns/{request_id}")
async def get_turn(task_id: str, request_id: str, request: Request) -> dict:
    """Return lightweight metadata for a single experience."""
    store = _store(request, read=True)
    exp = await _get_exp(store, task_id, request_id)
    return exp.to_dict()


@query_router.get("/tasks/{task_id}/turns/{request_id}/blob")
async def get_turn_blob(task_id: str, request_id: str, request: Request) -> Response:
    """Return the full experience as Trinity serialized bytes.

    Equivalent to ``Experience.serialize()``; deserializable via
    ``Experience.deserialize(...)`` or ``deserialize_many``.
    """
    store = _store(request, read=True)
    exp = await _get_exp(store, task_id, request_id)
    return Response(
        content=Experience.serialize(exp),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{request_id}.bin"'},
    )


@query_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, request: Request) -> dict:
    """Delete all experiences for a task."""
    store = _store(request, read=True)
    await store.delete_task(task_id)
    return {"task_id": task_id, "deleted": True}
