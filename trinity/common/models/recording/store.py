"""Pluggable storage backends for recorded experiences."""

import abc
from typing import Optional

from trinity.buffer.store import MemoryStore as BaseMemoryStore
from trinity.common.experience import Experience

REQUEST_ID_INFO_KEY = "request_id"
RECORD_KEY_INFO_KEY = "record_key"


class RecordStore(abc.ABC):
    """Abstract persistence interface for recorded experiences."""

    @abc.abstractmethod
    async def append_turn(self, exp: Experience) -> None:
        """Persist one completed experience."""

    @abc.abstractmethod
    async def update_reward_by_record_key(
        self, record_key: str, reward: float, run: int, task: str
    ) -> list[Experience]:
        """Set reward/run/task on every experience in the group, pop and return it."""

    @abc.abstractmethod
    async def get_record_experiences(self, record_key: str) -> list[Experience]:
        """Return all experiences for a record key, in insertion order."""

    @abc.abstractmethod
    async def get_request_experience(
        self, record_key: str, request_id: str
    ) -> Optional[Experience]:
        """Return a single experience, or None if not found."""

    @abc.abstractmethod
    async def list_records(self) -> list[str]:
        """Return all known record keys."""

    @abc.abstractmethod
    async def delete_record_experiences(self, record_key: str) -> None:
        """Drop all experiences for a record key."""

    @abc.abstractmethod
    async def delete_request_experience(self, record_key: str, request_id: str) -> bool:
        """Drop one experience by request id. Return True if one was deleted."""


class MemoryStore(BaseMemoryStore, RecordStore):
    """In-process store grouped by recording identity."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _group_key(exp: Experience) -> str:
        info = exp.info or {}
        record_key = info.get(RECORD_KEY_INFO_KEY)
        return record_key if record_key else exp.eid.suffix

    async def append_turn(self, exp: Experience) -> None:
        self.add(self._group_key(exp), [exp])

    async def update_reward_by_record_key(
        self, record_key: str, reward: float, run: int, task: str
    ) -> list[Experience]:
        if not self.get(record_key):
            return []
        self.update(
            key=record_key,
            reward=reward,
            info={"run": run, "task": task},
            sample_ids=None,
        )
        return self.remove(record_key)

    async def get_record_experiences(self, record_key: str) -> list[Experience]:
        return self.get(record_key)

    async def get_request_experience(
        self, record_key: str, request_id: str
    ) -> Optional[Experience]:
        for exp in self.get(record_key):
            info = exp.info or {}
            if info.get(REQUEST_ID_INFO_KEY) == request_id:
                return exp
        return None

    async def list_records(self) -> list[str]:
        return self.keys()

    async def delete_record_experiences(self, record_key: str) -> None:
        self.remove(record_key)

    async def delete_request_experience(self, record_key: str, request_id: str) -> bool:
        kept = []
        deleted = False
        for exp in self.get(record_key):
            info = exp.info or {}
            if info.get(REQUEST_ID_INFO_KEY) == request_id:
                deleted = True
            else:
                kept.append(exp)

        if deleted:
            if kept:
                self.overwrite(record_key, kept)
            else:
                await self.delete_record_experiences(record_key)
        return deleted
