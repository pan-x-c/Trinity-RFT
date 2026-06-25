"""Pluggable storage backends for recorded experiences."""

import abc
from collections import defaultdict
from typing import Optional

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
    async def get_task(self, record_key: str) -> list[Experience]:
        """Return all experiences for a record key, in insertion order."""

    @abc.abstractmethod
    async def get_turn(self, record_key: str, request_id: str) -> Optional[Experience]:
        """Return a single experience, or None if not found."""

    @abc.abstractmethod
    async def list_tasks(self) -> list[str]:
        """Return all known record keys."""

    @abc.abstractmethod
    async def delete_task(self, record_key: str) -> None:
        """Drop all experiences for a record key."""


class MemoryStore(RecordStore):
    """In-process store grouped by recording identity."""

    def __init__(self) -> None:
        self._records: dict[str, list[Experience]] = defaultdict(list)

    @staticmethod
    def _group_key(exp: Experience) -> str:
        record_key = exp.info.get(RECORD_KEY_INFO_KEY)
        return record_key if record_key else exp.eid.suffix

    async def append_turn(self, exp: Experience) -> None:
        self._records[self._group_key(exp)].append(exp)

    async def update_reward_by_record_key(
        self, record_key: str, reward: float, run: int, task: str
    ) -> list[Experience]:
        exps = self._records.pop(record_key, [])
        for exp in exps:
            exp.reward = reward
            exp.eid.run = run
            exp.eid.task = task
        return exps

    async def get_task(self, record_key: str) -> list[Experience]:
        return list(self._records.get(record_key, []))

    async def get_turn(self, record_key: str, request_id: str) -> Optional[Experience]:
        for exp in self._records.get(record_key, []):
            if exp.info.get(REQUEST_ID_INFO_KEY) == request_id:
                return exp
        return None

    async def list_tasks(self) -> list[str]:
        return list(self._records.keys())

    async def delete_task(self, record_key: str) -> None:
        self._records.pop(record_key, None)
