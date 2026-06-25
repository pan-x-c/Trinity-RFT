"""HTTP query/consume endpoints over recorded experiences.

Mounted on the main vLLM app (we own ``app`` in the bootstrap) via
``app.include_router(query_router)``. Routes are prefixed under ``/records`` to
avoid colliding with the OpenAI ``/v1/*`` surface — the OpenAI protocol is left
untouched.

Listing endpoints return ``Experience.to_dict()`` (lightweight metadata, no
tensor payload). A single-turn endpoint returns the full experience as Trinity
serialized bytes (``Experience.serialize()``) so it can be fed straight back
into Trinity pipelines.

``POST /records/consume_task`` is the consume path: it drains the recorder,
reward-stamps every experience in the matching record-key groups, pops them,
and returns the heavy experiences as ``Experience.serialize_many`` bytes — the
coordinator fans this out per rank at finalize time.
"""
from typing import List

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from trinity.common.experience import Experience
from trinity.common.models.vllm_patch.recording.recorder import Recorder
from trinity.common.models.vllm_patch.recording.store import RecordStore

#: Mounted on app.state by the bootstrap wiring in ``server.py``.
_STORE_ATTR = "trinity_record_store"
_RECORDER_ATTR = "trinity_recorder"

query_router = APIRouter(prefix="/records", tags=["trinity-recording"])


class _RecordKeyUpdate(BaseModel):
    """One record-key group to reward-stamp and consume."""

    record_key: str
    reward: float
    run: int = 0
    task: str = ""


class _ConsumeTaskRequest(BaseModel):
    updates: List[_RecordKeyUpdate]


def _store(request: Request) -> RecordStore:
    store = getattr(request.app.state, _STORE_ATTR, None)
    if store is None:
        raise HTTPException(status_code=503, detail="recording store not configured")
    return store


def _recorder(request: Request) -> Recorder:
    rec = getattr(request.app.state, _RECORDER_ATTR, None)
    if rec is None:
        raise HTTPException(status_code=503, detail="recorder not configured")
    return rec


async def _get_exp(store: RecordStore, record_key: str, request_id: str) -> Experience:
    exp = await store.get_turn(record_key, request_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experience not found")
    return exp


@query_router.get("/tasks")
async def list_tasks(request: Request) -> dict:
    """List all known record keys."""
    store = _store(request)
    return {"record_keys": await store.list_tasks()}


@query_router.get("/tasks/{record_key}")
async def get_task(record_key: str, request: Request) -> dict:
    """Return lightweight metadata for all experiences of a record key.

    Tensor payloads (tokens/logprobs/routed_experts) are omitted here to keep
    listing responses small; fetch the per-turn blob endpoint for full data.
    """
    store = _store(request)
    experiences = await store.get_task(record_key)
    return {"record_key": record_key, "turns": [e.to_dict() for e in experiences]}


@query_router.get("/tasks/{record_key}/turns/{request_id}")
async def get_turn(record_key: str, request_id: str, request: Request) -> dict:
    """Return lightweight metadata for a single experience."""
    store = _store(request)
    exp = await _get_exp(store, record_key, request_id)
    return exp.to_dict()


@query_router.get("/tasks/{record_key}/turns/{request_id}/blob")
async def get_turn_blob(record_key: str, request_id: str, request: Request) -> Response:
    """Return the full experience as Trinity serialized bytes.

    Equivalent to ``Experience.serialize()``; deserializable via
    ``Experience.deserialize(...)`` or ``deserialize_many``.
    """
    store = _store(request)
    exp = await _get_exp(store, record_key, request_id)
    return Response(
        content=Experience.serialize(exp),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{request_id}.bin"'},
    )


@query_router.delete("/tasks/{record_key}")
async def delete_task(record_key: str, request: Request) -> dict:
    """Delete all experiences for a record key."""
    store = _store(request)
    await store.delete_task(record_key)
    return {"record_key": record_key, "deleted": True}


@query_router.post("/consume_task")
async def consume_task(req: _ConsumeTaskRequest, request: Request) -> Response:
    """Reward-stamp and pop the experiences of the given record-key groups.

    Drains the recorder first (so every finished turn already made it into the
    store), then for each update sets ``reward``/``run``/``task`` on the whole
    matching group and pops it. Returns the union as Trinity serialized bytes
    (``Experience.serialize_many``), ready for the coordinator pipeline.

    A record_key absent from this rank yields no experiences (it lived on
    another rank); the coordinator fans this call out to every rank and merges.
    """
    store = _store(request)
    recorder = _recorder(request)
    # Ensure in-flight record tasks have been appended before we pop.
    await recorder.flush()

    exps: List[Experience] = []
    for update in req.updates:
        exps.extend(
            await store.update_reward_by_record_key(
                record_key=update.record_key,
                reward=update.reward,
                run=update.run,
                task=update.task,
            )
        )
    return Response(
        content=Experience.serialize_many(exps),
        media_type="application/octet-stream",
    )
