"""Pluggable storage backends for recorded experiences.

A ``RecordStore`` persists Trinity ``Experience`` objects in the vLLM API
server process. The only backend is ``MemoryStore`` — in-process, keyed by the
recording identity (``info["task_id"]`` = the API key / Ray-injected task id),
falling back to ``eid.suffix`` (the vLLM ``request_id``) when no identity was
supplied.

The consume side is ``update_reward_by_task_id``: it sets ``reward``/``run``/
``task`` on every experience in a task-id group, pops the group, and returns
it. This is the in-memory replacement for the old SQL ``HistoryRecorder``-
mediated join — the coordinator calls it (via ``/records/consume_task``) at
finalize time, so heavy experience bytes cross the network exactly once (store
→ coordinator pipeline) and never through Ray.

Keying: experiences are identified by ``eid.suffix`` (the vLLM ``request_id``)
for traceability, but **grouped** by ``info["task_id"]`` so a whole task's
worth of turns/samples can be reward-updated and consumed together.

Concurrency: ``append_turn`` is called from a single background flusher task;
``update_reward_by_task_id`` is called from the ``/records/consume_task`` HTTP
handler. Both run in the same asyncio loop, so the dict is single-writer-safe
across these two without a lock.
"""
import abc
from collections import defaultdict
from typing import Optional

from trinity.common.experience import Experience

#: Attribute carrying the vLLM request id on each experience's ``info`` dict.
_REQUEST_ID_INFO_KEY = "request_id"
#: Attribute carrying the recording identity on each experience's ``info``.
_TASK_ID_INFO_KEY = "task_id"


class RecordStore(abc.ABC):
    """Abstract persistence interface for recorded experiences."""

    @abc.abstractmethod
    async def append_turn(self, exp: Experience) -> None:
        """Persist one completed experience."""

    @abc.abstractmethod
    async def update_reward_by_task_id(
        self, task_id: str, reward: float, run: int, task: str
    ) -> list[Experience]:
        """Set reward/run/task on every experience in the group, pop and return it.

        Args:
            task_id: The recording identity (group key). When the recorded
                experience had no identity, this is its ``eid.suffix``.
            reward: Reward to stamp on every experience in the group.
            run: Run id to stamp on ``eid.run``.
            task: Task id to stamp on ``eid.task``.

        Returns:
            The (now reward-stamped) experiences of the group, in insertion
            order. Empty list if the group was absent.
        """

    @abc.abstractmethod
    async def get_task(self, task_id: str) -> list[Experience]:
        """Return all experiences for a task, in insertion order."""

    @abc.abstractmethod
    async def get_turn(self, task_id: str, request_id: str) -> Optional[Experience]:
        """Return a single experience, or None if not found."""

    @abc.abstractmethod
    async def list_tasks(self) -> list[str]:
        """Return all known task ids."""

    @abc.abstractmethod
    async def delete_task(self, task_id: str) -> None:
        """Drop all experiences for a task."""


class MemoryStore(RecordStore):
    """In-process store.

    Groups experiences by recording identity (``info["task_id"]``) when an API
    key / task id was supplied, otherwise each turn is keyed by its own
    ``eid.suffix`` (request_id) — so a missing identity never collapses
    distinct turns. ``get_turn`` resolves an individual turn by
    ``info["request_id"]``.

    Note: per-process under data-parallel serving — each API-server rank holds
    only the experiences it served. The coordinator fans out
    ``/records/consume_task`` to every rank and merges, so cross-rank
    aggregation happens at consume time, not in storage.
    """

    def __init__(self) -> None:
        # group key -> [experience, ...] in insertion order.
        self._records: dict[str, list[Experience]] = defaultdict(list)

    @staticmethod
    def _group_key(exp: Experience) -> str:
        session = exp.info.get(_TASK_ID_INFO_KEY)
        return session if session else exp.eid.suffix

    async def append_turn(self, exp: Experience) -> None:
        self._records[self._group_key(exp)].append(exp)

    async def update_reward_by_task_id(
        self, task_id: str, reward: float, run: int, task: str
    ) -> list[Experience]:
        exps = self._records.pop(task_id, [])
        for exp in exps:
            exp.reward = reward
            exp.eid.run = run
            exp.eid.task = task
        return exps

    async def get_task(self, task_id: str) -> list[Experience]:
        return list(self._records.get(task_id, []))

    async def get_turn(self, task_id: str, request_id: str) -> Optional[Experience]:
        for exp in self._records.get(task_id, []):
            if exp.info.get(_REQUEST_ID_INFO_KEY) == request_id:
                return exp
        return None

    async def list_tasks(self) -> list[str]:
        return list(self._records.keys())

    async def delete_task(self, task_id: str) -> None:
        self._records.pop(task_id, None)
