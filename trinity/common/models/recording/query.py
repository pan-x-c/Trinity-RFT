"""HTTP endpoints over recorded generation experiences."""

from typing import List

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from trinity.buffer.store import RecordStore
from trinity.common.experience import Experience
from trinity.common.models.recording.recorder import (
    TRINITY_RECORD_STORE_ATTR,
    TRINITY_RECORDER_ATTR,
    Recorder,
)

STORE_STATE_ATTR = TRINITY_RECORD_STORE_ATTR
RECORDER_STATE_ATTR = TRINITY_RECORDER_ATTR

query_router = APIRouter(prefix="/records", tags=["trinity-recording"])


class _RewardUpdateRequest(BaseModel):
    record_key: str
    reward: float
    info: dict | None = None
    sample_ids: List[str] | None = None


class _PrefixRequest(BaseModel):
    prefix: str


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
        if exp.eid.suffix == request_id:
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
        if exp.eid.suffix == request_id:
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


@query_router.post("/update_reward")
async def update_reward(req: _RewardUpdateRequest, request: Request) -> dict:
    store = _store(request)
    recorder = _recorder(request)
    await recorder.flush()
    if not store.get(req.record_key):
        return {"record_key": req.record_key, "updated": 0}
    store.update(
        key=req.record_key,
        reward=req.reward,
        info=req.info,
        sample_ids=req.sample_ids,
    )
    return {
        "record_key": req.record_key,
        "updated": (
            len(req.sample_ids) if req.sample_ids is not None else len(store.get(req.record_key))
        ),
    }


@query_router.post("/drain")
async def drain_records(req: _PrefixRequest, request: Request) -> Response:
    store = _store(request)
    recorder = _recorder(request)
    await recorder.flush()
    matched_keys = [
        key for key in store.keys() if key == req.prefix or key.startswith(f"{req.prefix}/")
    ]
    exps = store.remove(req.prefix)
    for key in matched_keys:
        recorder.forget_record(key)
    return Response(
        content=Experience.serialize_many(exps),
        media_type="application/octet-stream",
    )


@query_router.delete("")
async def delete_records(req: _PrefixRequest, request: Request) -> dict:
    store = _store(request)
    matched_keys = [
        key for key in store.keys() if key == req.prefix or key.startswith(f"{req.prefix}/")
    ]
    deleted = len(store.remove(req.prefix))
    for key in matched_keys:
        _forget_record(request, key)
    return {"prefix": req.prefix, "deleted": deleted}
