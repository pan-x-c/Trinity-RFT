"""HTTP query/update endpoints over recorded experiences."""

from typing import List

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from trinity.buffer.store import REQUEST_ID_INFO_KEY, RecordStore
from trinity.common.experience import Experience
from trinity.common.models.recording.recorder import (
    TRINITY_RECORD_STORE_ATTR,
    TRINITY_RECORDER_ATTR,
    Recorder,
)

STORE_STATE_ATTR = TRINITY_RECORD_STORE_ATTR
RECORDER_STATE_ATTR = TRINITY_RECORDER_ATTR

query_router = APIRouter(prefix="/records", tags=["trinity-recording"])


class _RecordUpdate(BaseModel):
    record_key: str
    reward: float


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


def _get_exp(store: RecordStore, record_key: str, request_id: str) -> Experience:
    for exp in store.get(record_key):
        info = exp.info or {}
        if info.get(REQUEST_ID_INFO_KEY) == request_id:
            return exp
    raise HTTPException(status_code=404, detail="experience not found")


def _forget_record(request: Request, record_key: str) -> None:
    rec = getattr(request.app.state, RECORDER_STATE_ATTR, None)
    if rec is not None:
        rec.forget_record(record_key)


@query_router.get("")
async def list_records(request: Request) -> dict:
    store = _store(request)
    return {"record_keys": store.keys()}


@query_router.get("/{record_key}")
async def get_record_experiences(record_key: str, request: Request) -> dict:
    store = _store(request)
    experiences = store.get(record_key)
    return {"record_key": record_key, "experiences": [e.to_dict() for e in experiences]}


@query_router.get("/{record_key}/request/{request_id}")
async def get_request_experience(record_key: str, request_id: str, request: Request) -> Response:
    store = _store(request)
    exp = _get_exp(store, record_key, request_id)
    return Response(
        content=Experience.serialize(exp),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{request_id}.bin"'},
    )


@query_router.delete("/{record_key}")
async def delete_record_experiences(record_key: str, request: Request) -> dict:
    store = _store(request)
    store.remove(record_key)
    _forget_record(request, record_key)
    return {"record_key": record_key, "deleted": True}


@query_router.delete("/{record_key}/request/{request_id}")
async def delete_request_experience(record_key: str, request_id: str, request: Request) -> dict:
    store = _store(request)
    kept = []
    deleted = False
    for exp in store.get(record_key):
        info = exp.info or {}
        if info.get(REQUEST_ID_INFO_KEY) == request_id:
            deleted = True
        else:
            kept.append(exp)
    if not deleted:
        raise HTTPException(status_code=404, detail="experience not found")
    if kept:
        store.overwrite(record_key, kept)
    else:
        store.remove(record_key)
    _forget_record(request, record_key)
    return {"record_key": record_key, "request_id": request_id, "deleted": True}


@query_router.post("/update_record")
async def update_record(req: _UpdateRecordRequest, request: Request) -> Response:
    store = _store(request)
    recorder = _recorder(request)
    await recorder.flush()

    exps: List[Experience] = []
    for update in req.updates:
        if not store.get(update.record_key):
            continue
        store.update(
            key=update.record_key,
            reward=update.reward,
            info=None,
            sample_ids=None,
        )
        exps.extend(store.remove(update.record_key))
        recorder.forget_record(update.record_key)
    return Response(
        content=Experience.serialize_many(exps),
        media_type="application/octet-stream",
    )
