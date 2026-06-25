"""Engine-agnostic background recorder for generated experiences."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Optional

from trinity.common.experience import Experience
from trinity.common.models.recording.context import skip_recording_ctx
from trinity.common.models.recording.store import RecordStore

MODEL_VERSION_ATTR = "trinity_model_version"

BuildExperiencesFn = Callable[..., Sequence[Experience]]


class Recorder:
    """Drains finished turns into a ``RecordStore`` from a background task.

    Engine-specific code supplies ``build_experiences``, which converts a
    finished engine output object into Trinity ``Experience`` instances.
    """

    def __init__(
        self,
        store: RecordStore,
        *,
        build_experiences: BuildExperiencesFn,
        enabled: bool,
        rank: int = 0,
        engine_client: Any = None,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.rank = rank
        self.engine_client = engine_client
        self._build_experiences = build_experiences
        self._queue: "asyncio.Queue[Optional[Experience]]" = asyncio.Queue()
        self._flusher: Optional[asyncio.Task] = None
        self._pending: "set[asyncio.Task]" = set()

    def start(self) -> None:
        """Start the background flusher. Idempotent."""
        if self._flusher is not None or not self.enabled:
            return
        self._flusher = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Drain in-flight + queued turns, then stop the flusher."""
        if self._flusher is None:
            return
        await self.flush()
        self._flusher.cancel()
        self._flusher = None

    def schedule_record(self, output: Any, record_key: Optional[str], **builder_kwargs) -> None:
        """Spawn and track a record task for a finished engine output."""
        task = asyncio.create_task(self._record(output, record_key, **builder_kwargs))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        """Wait until every in-flight record has been appended to the store."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        if self._flusher is not None:
            await self._queue.join()

    async def _record(self, output: Any, record_key: Optional[str], **builder_kwargs) -> None:
        if skip_recording_ctx.get():
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        model_version = getattr(self.engine_client, MODEL_VERSION_ATTR, None)
        exps = self._build_experiences(
            output,
            record_key,
            rank=self.rank,
            timestamp=timestamp,
            model_version=model_version,
            **builder_kwargs,
        )
        for exp in exps:
            await self._queue.put(exp)

    async def _flush_loop(self) -> None:
        while True:
            exp = await self._queue.get()
            try:
                if exp is None:
                    return
                await self._safe_append(exp)
            finally:
                self._queue.task_done()

    async def _safe_append(self, exp: Experience) -> None:
        try:
            await self.store.append_turn(exp)
        except Exception:
            logging.getLogger(__name__).exception(
                "recording store.append_turn failed for request %s",
                exp.info.get("request_id"),
            )
