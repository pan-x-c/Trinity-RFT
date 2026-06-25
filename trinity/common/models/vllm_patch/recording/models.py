# -*- coding: utf-8 -*-
"""Build a Trinity ``Experience`` from a finished vLLM ``RequestOutput``.

We record into Trinity's native ``Experience`` struct (see
``trinity.common.experience``) rather than a bespoke record, so captured data
drops straight into Trinity's RL/buffer pipeline without a conversion step.

Field mapping (captured ``RequestOutput`` fields -> ``Experience``):
  request_id        -> eid.suffix  (``EID(suffix=...)``; this is the msg_id the
                    proxy/openai client sees as ``response.id`` — the key the
                    proxy's ``HistoryRecorder.update_reward`` looks up by).
                    ``eid.task``/``run``/``reward`` are left default here and
                    assigned later by the proxy at ``/feedback`` time, matching
                    ``explorer/proxy/service.record_experience`` semantics.
  API key           -> info["task_id"]  (traceability only; not used as a key)
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

Plus bookkeeping (request_id / task_id / rank / timestamp / endpoint /
model_version) stashed in ``Experience.info`` so it round-trips with the
experience through serialize/deserialize.
"""
from typing import Any, Optional

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
) -> Optional[Experience]:
    """Build a Trinity ``Experience`` from a finished ``RequestOutput``.

    Args:
        output: A ``RequestOutput`` with ``finished == True``.
        task_id: From the request API key; stored in ``info`` for traceability
            only.
            Not used as the storage key — ``eid.suffix`` is, so a missing
            API key never drops a turn.
        rank: Data-parallel serving rank.
        timestamp: UTC ISO-8601 string (caller-stamped to keep this pure).
        endpoint: Which OpenAI endpoint served the turn (best-effort).
        model_version: Checkpoint version the serving policy was at; stamped
            into ``info`` for RL attribution (read in-actor by the recorder).

    Returns:
        A populated ``Experience``, or None if the turn is degenerate (no
        prompt or no response tokens) and cannot form a valid experience.
    """
    request_id = output.request_id
    # Key by the request id (= the OpenAI response ``id`` / proxy msg_id) so the
    # proxy's HistoryRecorder.update_reward can find this row at feedback time.
    # task/run/reward are intentionally left default — the proxy assigns them.

    prompt_token_ids = list(output.prompt_token_ids or [])

    completion = output.outputs[0] if output.outputs else None
    if completion is None:
        return None
    response_token_ids = list(completion.token_ids or [])

    # A valid single-turn experience needs both a prompt and a response;
    # Experience.__init__ asserts len(tokens) > prompt_length otherwise.
    if not prompt_token_ids or not response_token_ids:
        return None

    tokens = prompt_token_ids + response_token_ids
    prompt_length = len(prompt_token_ids)

    chosen_logprobs = _extract_chosen_logprobs(completion.logprobs, response_token_ids)
    routed_experts = completion.routed_experts

    info = {
        "request_id": request_id,
        "task_id": task_id,
        "rank": rank,
        "timestamp": timestamp,
        "endpoint": endpoint,
        "model_version": model_version,
    }

    return Experience(
        eid=EID(suffix=request_id),
        tokens=tokens,
        logprobs=chosen_logprobs,
        prompt_length=prompt_length,
        routed_experts=routed_experts,
        prompt_text=output.prompt,
        response_text=completion.text,
        info=info,
    )
