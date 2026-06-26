"""HTTP query/update endpoints over recorded experiences."""

from typing import List

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from trinity.common.experience import Experience
from trinity.common.models.recording.recorder import (
    TRINITY_RECORD_STORE_ATTR,
    TRINITY_RECORDER_ATTR,
    Recorder,
)
from trinity.common.models.recording.store import RecordStore

STORE_STATE_ATTR = TRINITY_RECORD_STORE_ATTR
RECORDER_STATE_ATTR = TRINITY_RECORDER_ATTR

query_router = APIRouter(prefix="/records", tags=["trinity-recording"])


class _RecordUpdate(BaseModel):
    record_key: str
    reward: float
    run: int = 0
    task: str = ""


class _UpdateRecordRequest(BaseModel):
    updates: List[_RecordUpdate]


def _store(request: Request) -> RecordStore:
    store = getattr(request.app.state, STORE_STATE_ATTR, None)
    if store is None:
        raise HTTPException(status_code=503, detail="recording store not configured")
    return store


def _recorder(request: Request) -> Recorder:
    rec = getattr(request.app.state, RECORDER_STATE_ATTR, None)
    if rec is None:
        raise HTTPException(status_code=503, detail="recorder not configured")
    return rec


async def _get_exp(store: RecordStore, record_key: str, request_id: str) -> Experience:
    exp = await store.get_request_experience(record_key, request_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experience not found")
    return exp


@query_router.get("")
async def list_records(request: Request) -> dict:
    store = _store(request)
    return {"record_keys": await store.list_records()}


@query_router.get("/{record_key}")
async def get_record_experiences(record_key: str, request: Request) -> dict:
    store = _store(request)
    experiences = await store.get_record_experiences(record_key)
    return {"record_key": record_key, "experiences": [e.to_dict() for e in experiences]}


@query_router.get("/{record_key}/request/{request_id}")
async def get_request_experience(record_key: str, request_id: str, request: Request) -> Response:
    store = _store(request)
    exp = await _get_exp(store, record_key, request_id)
    return Response(
        content=Experience.serialize(exp),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{request_id}.bin"'},
    )


@query_router.delete("/{record_key}")
async def delete_record_experiences(record_key: str, request: Request) -> dict:
    store = _store(request)
    await store.delete_record_experiences(record_key)
    return {"record_key": record_key, "deleted": True}


@query_router.delete("/{record_key}/request/{request_id}")
async def delete_request_experience(record_key: str, request_id: str, request: Request) -> dict:
    store = _store(request)
    deleted = await store.delete_request_experience(record_key, request_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="experience not found")
    return {"record_key": record_key, "request_id": request_id, "deleted": True}


@query_router.post("/update_record")
async def update_record(req: _UpdateRecordRequest, request: Request) -> Response:
    store = _store(request)
    recorder = _recorder(request)
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
