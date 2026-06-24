"""Engine-level wrap that forces top-k logprobs and records finished turns.

This is the heart of the recording patch. It follows the same instance-level
wrap pattern as ``api_patch_v17.patch_vllm_reasoning_content_alias``:
``functools.wraps`` + a ``__patched_*__`` guard attribute to stay idempotent.

Why wrap ``engine_client.generate`` instead of the serving layer?
  * The serving layer (``OpenAIServingChat``/``OpenAIServingCompletion``) is
    what decides streaming vs non-streaming and what fields to emit. vLLM does
    NOT put ``routed_experts`` into streaming responses (the streaming choice
    schemas omit it), so capturing at the HTTP layer misses it.
  * ``RequestOutput`` / ``CompletionOutput`` carry the full data regardless of
    streaming mode: ``prompt_token_ids``, ``token_ids``, ``logprobs``,
    ``routed_experts`` (raw ndarray). Wrapping at the engine boundary captures
    all four uniformly for chat / completion / responses endpoints.
  * Forcing ``sampling_params.logprobs`` here only affects engine-internal
    computation — the client response is unchanged unless the client itself
    requested logprobs. Recording stays transparent.
"""
import asyncio
import functools
import logging
from typing import Optional

from trinity.common.experience import Experience
from trinity.common.models.vllm_patch.recording.context import task_id_ctx
from trinity.common.models.vllm_patch.recording.models import build_experience
from trinity.common.models.vllm_patch.recording.store import RecordStore

#: Guard attribute marking the wrapped generate, mirroring api_patch_v17 style.
_PATCHED_FLAG = "__patched_engine_recording__"
#: Instance attribute on the AsyncLLM engine_client holding the current serving
#: checkpoint version. Mirrored by ``VLLMModel.sync_model_weights`` (and at
#: engine creation); read live here so each experience is attributed to the
#: right policy without a launch-time parameter.
_MODEL_VERSION_ATTR = "trinity_model_version"


def patch_engine_for_recording(
    engine_client,
    recorder: "Recorder",
    logger: logging.Logger,
) -> None:
    """Wrap ``engine_client.generate`` in place to record finished turns.

    Instance-level: only this server's engine_client is affected, the global
    class is untouched. Must run before ``init_app_state`` stores the engine
    reference into the serving objects (they hold the same object, so the wrap
    is inherited).

    Args:
        engine_client: The AsyncLLM instance passed into the bootstrap.
        recorder: The ``Recorder`` that will persist turns.
        logger: Logger for the idempotency/confirmation message.

    Raises:
        RuntimeError: If ``engine_client.generate`` is missing (unexpected
            vLLM version drift).
    """
    current = getattr(engine_client, "generate", None)
    if current is None:
        raise RuntimeError("vLLM patch failed: engine_client.generate not found")
    if getattr(current, _PATCHED_FLAG, False):
        return

    @functools.wraps(current)
    async def _patched_generate(*args, **kwargs):
        # generate(prompt, sampling_params, request_id, *, ...).
        # ``engine_client.generate`` assigned as an instance attribute is NOT
        # bound, so ``self`` is absent and args map 1:1 to the protocol.
        sampling_params = kwargs.get("sampling_params")
        if sampling_params is None and len(args) >= 2:
            sampling_params = args[1]

        if recorder.enabled and sampling_params is not None:
            desired = recorder.topk
            cur = sampling_params.logprobs
            sampling_params.logprobs = max(cur, desired) if cur is not None else desired

        last = None
        # ``current`` is the original *bound* method captured pre-wrap, so it
        # still resolves ``self`` correctly. Yields RequestOutput unchanged.
        async for out in current(*args, **kwargs):
            last = out
            yield out

        if recorder.enabled and last is not None and getattr(last, "finished", False):
            # Recover task id from the request's async context (set by
            # SessionMiddleware). None when the client omitted X-Session-ID;
            # the recorder then falls back to request_id.
            task_id = task_id_ctx.get()
            # Offload heavy serialization off the response critical path.
            asyncio.create_task(recorder.record(last, task_id))

    setattr(_patched_generate, _PATCHED_FLAG, True)
    engine_client.generate = _patched_generate
    logger.info("Patched vLLM engine_client.generate for generation recording")


class Recorder:
    """Drains finished turns into a ``RecordStore`` from a background task.

    Putting records into an ``asyncio.Queue`` and flushing from a single worker
    keeps the response path cheap (record == one ``queue.put``) and serializes
    expensive payloads (ndarray -> .npy, json) off the serving hot loop.
    """

    def __init__(
        self,
        store: RecordStore,
        *,
        topk: int,
        enabled: bool,
        rank: int = 0,
        engine_client=None,
    ) -> None:
        self.store = store
        self.topk = topk
        self.enabled = enabled
        self.rank = rank
        # The engine_client is the same AsyncLLM instance VLLMModel updates in
        # sync_model_weights (``.trinity_model_version``), so we read the live
        # checkpoint version off it at record time.
        self.engine_client = engine_client
        self._queue: "asyncio.Queue[Optional[Experience]]" = asyncio.Queue()
        self._flusher: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the background flusher. Idempotent."""
        if self._flusher is not None or not self.enabled:
            return
        self._flusher = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Cancel the flusher and drain remaining queued turns."""
        if self._flusher is None:
            return
        self._flusher.cancel()
        self._flusher = None
        # Drain anything already queued so we don't lose in-flight turns.
        while not self._queue.empty():
            exp = self._queue.get_nowait()
            if exp is not None:
                await self._safe_append(exp)

    async def record(self, output, task_id: Optional[str]) -> None:
        """Enqueue a finished ``RequestOutput`` for recording as an Experience.

        Args:
            output: A finished ``RequestOutput``.
            task_id: From ``task_id_ctx``; stored in ``info`` for traceability.
        """
        # Stamp now (real runtime, not a workflow sandbox): permitted here.
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        # Read the live checkpoint version the actor mirrors onto the engine.
        model_version = getattr(self.engine_client, _MODEL_VERSION_ATTR, None)
        exp = build_experience(
            output,
            task_id,
            rank=self.rank,
            timestamp=timestamp,
            model_version=model_version,
        )
        if exp is None:
            # Degenerate turn (no prompt/response) — nothing to record.
            return
        await self._queue.put(exp)

    async def _flush_loop(self) -> None:
        while True:
            exp = await self._queue.get()
            if exp is None:
                # Sentinel for graceful shutdown.
                return
            await self._safe_append(exp)

    async def _safe_append(self, exp: Experience) -> None:
        try:
            await self.store.append_turn(exp)
        except Exception:
            # Never let a storage hiccup crash the flusher loop.
            logging.getLogger(__name__).exception(
                "recording store.append_turn failed for request %s",
                exp.info.get("request_id"),
            )
