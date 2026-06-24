"""Pluggable storage backends for recorded experiences.

A ``RecordStore`` persists Trinity ``Experience`` objects. Backends:

* ``MemoryStore`` — in-process, request/session-keyed; for standalone runs and
  the ``/records`` debug endpoints.
* ``SqlStore`` — delegates to ``trinity.explorer.proxy.recorder.HistoryRecorder``
  so the in-vLLM recorder writes to the *same* SQL table the explorer proxy
  reads (``proxy_history``). This is the online-RL path: experiences written
  here by the vLLM process are later picked up by the proxy's
  ``update_reward``/``submit_experiences`` via the shared ``msg_id`` key.

Keying: experiences are identified by ``eid.suffix`` (the vLLM ``request_id``,
== the OpenAI ``response.id`` == the proxy ``msg_id``). ``eid.task``/``run``/
``reward`` are assigned by the proxy at feedback time, not here.

Concurrency: ``append_turn`` is called from a single background flusher task;
the async signatures keep the door open for I/O-bound backends.
"""
import abc
from collections import defaultdict
from typing import Optional

from trinity.common.experience import Experience

#: Attribute carrying the vLLM request id on each experience's ``info`` dict.
_REQUEST_ID_INFO_KEY = "request_id"
#: Attribute carrying the task id (X-Session-ID) on each experience's ``info``.
_TASK_ID_INFO_KEY = "task_id"


class RecordStore(abc.ABC):
    """Abstract persistence interface for recorded experiences."""

    @abc.abstractmethod
    async def append_turn(self, exp: Experience) -> None:
        """Persist one completed experience."""

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

    Groups experiences by session (``info["task_id"]``) when an X-Session-ID
    was supplied, otherwise each turn is keyed by its own ``eid.suffix``
    (request_id) — so a missing session header never collapses distinct turns.
    ``get_turn`` resolves an individual turn by ``info["request_id"]``.

    Note: per-process under data-parallel serving — each API-server rank holds
    only the experiences it served. For cross-rank aggregation, use ``SqlStore``.
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


class SqlStore(RecordStore):
    """SQL-backed store sharing the explorer proxy's ``proxy_history`` table.

    Writes go through ``HistoryRecorder.record_history`` (which ``prepare()``s
    the engine on first use and maps ``eid.suffix`` -> ``msg_id``). The explorer
    proxy's own ``HistoryRecorder`` instance reads/updates the same rows for
    ``/feedback`` and ``/commit``, so the in-vLLM recorder and the proxy share
    one table by ``db_url`` + ``table_name``.

    Reads (``get_task``/``get_turn``/``list_tasks``/``delete_task``) are NOT
    implemented here: in the online-RL setup the proxy owns the read/consume
    side. The ``/records`` query endpoints surface this as 503 when this backend
    is active.
    """

    #: Marks that this backend does not serve the ``/records`` read endpoints.
    supports_reads = False

    def __init__(self, db_url: str, table_name: str) -> None:
        # Imported lazily so the vLLM process only pulls in the SQL/explorer
        # stack when this backend is actually selected.
        from trinity.explorer.proxy.recorder import HistoryRecorder

        self._recorder = HistoryRecorder(db_url=db_url, table_name=table_name)

    async def append_turn(self, exp: Experience) -> None:
        # record_history() calls prepare() on first use; serializes the
        # experience into the blob column and writes meta keyed by msg_id.
        await self._recorder.record_history([exp])

    async def get_task(self, task_id: str) -> list[Experience]:
        raise NotImplementedError("SqlStore reads are served by the explorer proxy; use /feedback")

    async def get_turn(self, task_id: str, request_id: str) -> Optional[Experience]:
        raise NotImplementedError("SqlStore reads are served by the explorer proxy; use /feedback")

    async def list_tasks(self) -> list[str]:
        raise NotImplementedError("SqlStore reads are served by the explorer proxy; use /feedback")

    async def delete_task(self, task_id: str) -> None:
        raise NotImplementedError("SqlStore reads are served by the explorer proxy; use /feedback")
