# -*- coding: utf-8 -*-
"""Build Trinity ``Experience`` objects from a finished vLLM ``RequestOutput``.

We record into Trinity's native ``Experience`` struct (see
``trinity.common.experience``) rather than a bespoke record, so captured data
drops straight into Trinity's RL/buffer pipeline without a conversion step.

A single ``RequestOutput`` may carry multiple completions (``n > 1``); we emit
one ``Experience`` per completion so no sample is lost.

Field mapping (captured ``RequestOutput`` fields -> ``Experience``):
  request_id        -> eid.suffix  (``EID(suffix=...)``; the vLLM engine request
                    id == the OpenAI ``response.id``. Kept for traceability;
                    ``eid.task``/``run``/``reward`` are left default here and
                    assigned by ``MemoryStore.update_reward_by_task_id`` at
                    consume time.)
  API key / task id -> info["task_id"]  (the recording identity; **the group
                    key** the MemoryStore batches experiences by, so a whole
                    task's samples/turns are reward-updated and consumed
                    together. Falls back to ``eid.suffix`` when absent.)
  sample index      -> info["sample_index"]  (position within the n-completion
                    set; orders samples/turns inside a task-id group)
  prompt_token_ids  -> tokens (prompt portion) + prompt_length
  response_token_ids-> tokens (response portion)
  logprobs          -> Experience.logprobs  -- but ONLY the *chosen* token's
                    logprob per position (flat ``[resp_length]`` tensor), per
                    the RL convention. vLLM's ``CompletionOutput.logprobs`` is a
                    top-k structure per position; we look up the actually-sampled
                    token id and take its ``.logprob``.
  routed_experts   -> Experience.routed_experts (uint8 tensor, verbatim)
  model_version    -> info["model_version"]  (which checkpoint policy served the
                    turn; read in-actor by the recorder's provider)

Plus bookkeeping (request_id / task_id / sample_index / rank / timestamp /
endpoint / model_version) stashed in ``Experience.info`` so it round-trips
with the experience through serialize/deserialize.
"""
from typing import Any, List, Optional

from trinity.common.experience import EID, Experience


def _extract_chosen_logprobs(
    sample_logprobs: Any,
    response_token_ids: list[int],
) -> Optional[list[float]]:
    """Pull the sampled token's logprob at each response position.

    vLLM exposes ``CompletionOutput.logprobs`` as either a list of
    ``dict[int, Logprob]`` or the ``FlatLogprobs`` container; both support
    positional indexing returning ``dict[int, Logprob]`` for that position, so
    we treat them uniformly.

    Returns a flat ``[resp_length]`` list of floats, or None when logprobs were
    not requested/computed.

    Note: the sampled token is *always* present at each position's dict. vLLM
    force-includes it as column 0 of the reported set
    (``vllm/v1/worker/gpu/sample/logprob.py:compute_topk_logprobs``), so a
    request with ``sampling_params.logprobs = N`` reports ``{sampled} ∪
    top-N`` — the chosen token is reported even when it ranks beyond N in the
    model's distribution. There is therefore no "sampled token absent from
    top-k" case to handle here: ``pos[tid]`` always resolves, and a length
    mismatch between ``sample_logprobs`` and ``response_token_ids`` cannot
    occur in normal operation (both are indexed per generated token).

    Args:
        sample_logprobs: ``CompletionOutput.logprobs`` (may be None).
        response_token_ids: The generated token ids.

    Returns:
        Flat list of chosen-token logprobs, or None.
    """
    if not sample_logprobs:
        return None
    # One entry per generated token; sampled token is force-included per the
    # note above, so a direct lookup per position is always well-defined.
    return [float(sample_logprobs[i][tid].logprob) for i, tid in enumerate(response_token_ids)]


def build_experience(
    output: Any,
    task_id: Optional[str],
    *,
    rank: int,
    timestamp: str,
    endpoint: str = "unknown",
    model_version: Optional[int] = None,
) -> List[Experience]:
    """Build Trinity ``Experience`` objects from a finished ``RequestOutput``.

    One experience per completion (``output.outputs``), so ``n > 1`` sampling
    is captured in full. Each experience shares ``eid.suffix = request_id`` and
    ``info["task_id"] = task_id`` (the group key); ``info["sample_index"]``
    distinguishes samples within the group.

    Args:
        output: A ``RequestOutput`` with ``finished == True``.
        task_id: The recording identity (API key / Ray-injected task id);
            stored in ``info["task_id"]`` and used as the MemoryStore group key.
        rank: Data-parallel serving rank.
        timestamp: UTC ISO-8601 string (caller-stamped to keep this pure).
        endpoint: Which OpenAI endpoint served the turn (best-effort).
        model_version: Checkpoint version the serving policy was at; stamped
            into ``info`` for RL attribution (read in-actor by the recorder).

    Returns:
        One ``Experience`` per non-degenerate completion. Empty list if the
        request had no prompt or no completion with response tokens.
    """
    request_id = output.request_id
    # eid.suffix = request_id for traceability; task/run/reward are left
    # default and assigned by MemoryStore.update_reward_by_task_id at consume.

    prompt_token_ids = list(output.prompt_token_ids or [])
    if not prompt_token_ids:
        return []

    completions = list(output.outputs or [])
    if not completions:
        return []

    experiences: List[Experience] = []
    for sample_index, completion in enumerate(completions):
        response_token_ids = list(completion.token_ids or [])
        # A valid single-turn experience needs both a prompt and a response;
        # Experience.__init__ asserts len(tokens) > prompt_length otherwise.
        if not response_token_ids:
            continue

        tokens = prompt_token_ids + response_token_ids
        prompt_length = len(prompt_token_ids)

        chosen_logprobs = _extract_chosen_logprobs(completion.logprobs, response_token_ids)
        routed_experts = completion.routed_experts

        info = {
            "request_id": request_id,
            "task_id": task_id,
            "sample_index": sample_index,
            "rank": rank,
            "timestamp": timestamp,
            "endpoint": endpoint,
            "model_version": model_version,
        }

        experiences.append(
            Experience(
                eid=EID(suffix=request_id),
                tokens=tokens,
                logprobs=chosen_logprobs,
                prompt_length=prompt_length,
                routed_experts=routed_experts,
                prompt_text=output.prompt,
                response_text=completion.text,
                info=info,
            )
        )
    return experiences
