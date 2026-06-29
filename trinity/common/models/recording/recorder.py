"""Engine-agnostic background recorder for generated experiences."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Optional

import torch

from trinity.buffer.store import RecordStore, default_sample_id_getter, get_record_key
from trinity.common.experience import Experience
from trinity.common.models.recording.context import skip_recording_ctx

MODEL_VERSION_ATTR = "trinity_model_version"
TRINITY_RECORDER_ATTR = "trinity_recorder"
TRINITY_RECORD_STORE_ATTR = "trinity_record_store"

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
        merge_prefix_experiences: bool = True,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.rank = rank
        self.engine_client = engine_client
        self.merge_prefix_experiences = merge_prefix_experiences
        self._build_experiences = build_experiences
        self._queue: "asyncio.Queue[Optional[Experience]]" = asyncio.Queue()
        self._flusher: Optional[asyncio.Task] = None
        self._pending: "set[asyncio.Task]" = set()
        self._merge_heads: dict[str, dict[tuple[str, Any], Experience]] = {}

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
            record_key = get_record_key(exp)
            if self.merge_prefix_experiences and self._merge_or_append(record_key, exp):
                return
            self.store.add(record_key, [exp])
            if self.merge_prefix_experiences:
                self._remember_merge_head(record_key, exp)
        except Exception:
            logging.getLogger(__name__).exception(
                "recording store write failed for request %s",
                exp.info.get("request_id"),
            )

    def _merge_or_append(self, record_key: str, exp: Experience) -> bool:
        stream_key = _sample_stream_key(exp)
        heads = self._merge_heads.setdefault(record_key, {})
        candidate = heads.get(stream_key)
        if candidate is None:
            candidate = _find_longest_prefix_experience(self.store.get(record_key), exp)
        elif not _is_strict_token_prefix(candidate.tokens, exp.tokens):
            return False
        if candidate is None:
            return False

        old_sample_id = default_sample_id_getter(candidate)
        merged = _merge_prefix_experiences(candidate, exp)
        try:
            self.store.replace(record_key, old_sample_id, merged)
        except KeyError:
            heads.pop(stream_key, None)
            return False
        heads[stream_key] = merged
        return True

    def _remember_merge_head(self, record_key: str, exp: Experience) -> None:
        self._merge_heads.setdefault(record_key, {})[_sample_stream_key(exp)] = exp

    def forget_record(self, record_key: str) -> None:
        self._merge_heads.pop(record_key, None)


def _find_longest_prefix_experience(
    existing: Sequence[Experience],
    exp: Experience,
) -> Optional[Experience]:
    best_candidate = None
    best_length = -1
    for candidate in existing:
        candidate_length = len(candidate.tokens)
        if candidate_length <= best_length:
            continue
        if not _same_sample_stream(candidate, exp):
            continue
        if _is_strict_token_prefix(candidate.tokens, exp.tokens):
            best_candidate = candidate
            best_length = candidate_length
    return best_candidate


def _same_sample_stream(left: Experience, right: Experience) -> bool:
    return _sample_stream_key(left) == _sample_stream_key(right)


def _sample_stream_key(exp: Experience) -> tuple[str, Any]:
    info = exp.info or {}
    sample_index = info.get("sample_index")
    if sample_index is not None:
        return ("sample_index", sample_index)

    sample_id = info.get("sample_id")
    if sample_id is not None:
        return ("sample_id", sample_id)

    request_id = info.get("request_id")
    if isinstance(request_id, str):
        _, sep, suffix = request_id.rpartition(":")
        if sep and suffix.isdigit():
            return ("request_id_sample_index", int(suffix))

    return ("default", 0)


def _is_strict_token_prefix(prefix: torch.Tensor, tokens: torch.Tensor) -> bool:
    prefix_len = len(prefix)
    if prefix_len == 0 or prefix_len >= len(tokens):
        return False
    if prefix.device == tokens.device:
        return bool(torch.equal(prefix.detach(), tokens[:prefix_len].detach()))
    return bool(torch.equal(prefix.detach().cpu(), tokens[:prefix_len].detach().cpu()))


def _merge_prefix_experiences(prefix_exp: Experience, final_exp: Experience) -> Experience:
    prefix_len = len(prefix_exp.tokens)
    final_prompt_length = final_exp.prompt_length
    if final_prompt_length < prefix_len:
        final_prompt_length = prefix_len
    gap_len = final_prompt_length - prefix_len
    final_response_len = len(final_exp.tokens) - final_prompt_length

    prefix_action_mask = _response_action_mask(prefix_exp)
    final_source_mask = _response_action_mask(final_exp)
    final_action_mask = (
        final_source_mask[-final_response_len:] if final_response_len else final_source_mask[:0]
    )
    if gap_len:
        action_mask = torch.cat(
            [
                prefix_action_mask,
                torch.zeros(gap_len, dtype=torch.bool, device=prefix_action_mask.device),
                final_action_mask,
            ]
        )
    else:
        action_mask = torch.cat([prefix_action_mask, final_action_mask])

    logprobs = _merge_logprobs(prefix_exp, final_exp, gap_len, final_response_len)
    routed_experts = _merge_routed_experts(prefix_exp, final_exp, gap_len, final_response_len)
    info = _merge_info(prefix_exp, final_exp)

    return Experience(
        eid=final_exp.eid,
        tokens=final_exp.tokens,
        logprobs=logprobs,
        reward=final_exp.reward,
        token_level_reward=final_exp.token_level_reward,
        advantages=final_exp.advantages,
        returns=final_exp.returns,
        truncate_status=final_exp.truncate_status or prefix_exp.truncate_status,
        info=info,
        metrics=final_exp.metrics,
        prompt_length=prefix_exp.prompt_length,
        response_text=final_exp.response_text,
        prompt_text=prefix_exp.prompt_text,
        action_mask=action_mask,
        messages=final_exp.messages or prefix_exp.messages,
        tools=final_exp.tools or prefix_exp.tools,
        multi_modal_inputs=final_exp.multi_modal_inputs,
        teacher_logprobs=final_exp.teacher_logprobs,
        routed_experts=routed_experts,
        custom_fields=final_exp.custom_fields,
    )


def _response_action_mask(exp: Experience) -> torch.Tensor:
    response_len = len(exp.tokens) - exp.prompt_length
    if exp.action_mask is None:
        return torch.ones(response_len, dtype=torch.bool)
    return exp.action_mask.to(dtype=torch.bool)


def _merge_logprobs(
    prefix_exp: Experience,
    final_exp: Experience,
    gap_len: int,
    final_response_len: int,
) -> Optional[torch.Tensor]:
    if prefix_exp.logprobs is None or final_exp.logprobs is None:
        return None
    parts = [prefix_exp.logprobs]
    if gap_len:
        parts.append(
            torch.zeros(
                gap_len,
                dtype=prefix_exp.logprobs.dtype,
                device=prefix_exp.logprobs.device,
            )
        )
    parts.append(
        final_exp.logprobs[-final_response_len:] if final_response_len else final_exp.logprobs[:0]
    )
    return torch.cat(parts)


def _merge_routed_experts(
    prefix_exp: Experience,
    final_exp: Experience,
    gap_len: int,
    final_response_len: int,
) -> Optional[torch.Tensor]:
    prefix_routed = _response_routed_experts(prefix_exp)
    final_routed = _response_routed_experts(final_exp)
    if prefix_routed is None or final_routed is None:
        return None
    parts = [prefix_routed]
    if gap_len:
        parts.append(
            torch.zeros(
                (gap_len, *prefix_routed.shape[1:]),
                dtype=prefix_routed.dtype,
                device=prefix_routed.device,
            )
        )
    parts.append(final_routed[-final_response_len:] if final_response_len else final_routed[:0])
    return torch.cat(parts, dim=0)


def _response_routed_experts(exp: Experience) -> Optional[torch.Tensor]:
    routed = exp.routed_experts
    if routed is None:
        return None
    response_len = len(exp.tokens) - exp.prompt_length
    if len(routed) == response_len:
        return routed
    # Full-sequence routing is aligned to next-token predictions:
    # token i uses routing row i - 1, so response tokens start at prompt_length - 1.
    if len(routed) == len(exp.tokens) - 1:
        return routed[exp.prompt_length - 1 :]
    return None


def _merge_info(prefix_exp: Experience, final_exp: Experience) -> dict:
    info = dict(final_exp.info or {})

    merged_request_ids = list((prefix_exp.info or {}).get("merged_request_ids") or [])
    prefix_request_id = (prefix_exp.info or {}).get("request_id")
    if prefix_request_id is not None and prefix_request_id not in merged_request_ids:
        merged_request_ids.append(prefix_request_id)
    final_request_id = (final_exp.info or {}).get("request_id")
    if final_request_id is not None and final_request_id not in merged_request_ids:
        merged_request_ids.append(final_request_id)
    if merged_request_ids:
        info["merged_request_ids"] = merged_request_ids

    merged_sample_ids = list((prefix_exp.info or {}).get("merged_sample_ids") or [])
    for sample_id in (default_sample_id_getter(prefix_exp), default_sample_id_getter(final_exp)):
        if sample_id not in merged_sample_ids:
            merged_sample_ids.append(sample_id)
    info["merged_sample_ids"] = merged_sample_ids
    info["merged_turn_count"] = int((prefix_exp.info or {}).get("merged_turn_count") or 1) + 1
    return info
