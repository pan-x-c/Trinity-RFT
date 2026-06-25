# -*- coding: utf-8 -*-
"""Recording configuration carried explicitly through the launch chain.

Instead of env vars or attributes bolted onto the engine instance, the static
recording config (db url / table / top-k) is passed as an explicit parameter
into ``run_api_server_with_recording``. ``get_api_server`` builds one from
``InferenceModelConfig`` when ``enable_recording`` is on.

The *dynamic* checkpoint version is NOT here: it changes at runtime
(``sync_model_weights``), so the recorder reads it live off the engine
instance attribute ``trinity_model_version`` (mirrored by VLLMModel).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RecordingConfig:
    """Static configuration for the in-vLLM recorder.

    Attributes:
        db_url: SQL db url the recorder writes to (shared with the explorer
            proxy's HistoryRecorder). When None, the recorder falls back to an
            in-process MemoryStore (no cross-process visibility).
        table: SQL table name (default ``proxy_history``, matching the proxy).
        topk: How many top-k logprobs the engine computes per generated token.
            Only the chosen token's logprob is stored, and vLLM force-includes
            the sampled token, so 1 suffices by default.
    """

    db_url: Optional[str] = None
    table: str = "proxy_history"
    topk: int = 1
