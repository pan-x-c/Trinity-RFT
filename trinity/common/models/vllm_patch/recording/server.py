"""Self-contained bootstrap that copies api_patch_v17.py's server lifecycle
and additionally wires in generation recording.

This module deliberately mirrors ``api_patch_v17.py`` so it can be used as a
drop-in alternative: point your launcher at
``trinity.common.models.vllm_patch.recording.run_api_server_with_recording``
and you get the standard vLLM OpenAI server *plus* generation recording, with
no edits to vLLM source or to ``api_patch_v17.py``.

Recording wiring (all applied between ``build_app`` and ``serve_http`` because
we own both ``app`` and ``engine_client`` at that point):
  1. ``patch_engine_for_recording`` — instance-level wrap of
     ``engine_client.generate`` to force top-k logprobs and record finished
     ``RequestOutput`` (covers chat/completion/responses, streaming and not).
  2. ``RecordingIdentityMiddleware`` — in-process ASGI middleware reading
     ``Authorization: Bearer <api_key>`` into a contextvar.
  3. ``query_router`` — ``/records/*`` endpoints for later analysis.

Only for vllm versions >= 0.17.0.
"""
import asyncio
import functools
import logging
from typing import Optional

import vllm
import vllm.envs as envs
from packaging.version import parse as parse_version
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    create_server_unix_socket,
    init_app_state,
    validate_api_server_args,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.serve.utils.api_utils import log_non_default_args
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

from trinity.common.models.vllm_patch import get_vllm_version
from trinity.common.models.vllm_patch.recording.config import RecordingConfig
from trinity.common.models.vllm_patch.recording.context import (
    RecordingIdentityMiddleware,
)
from trinity.common.models.vllm_patch.recording.query import query_router
from trinity.common.models.vllm_patch.recording.recorder import (
    Recorder,
    patch_engine_for_recording,
)
from trinity.common.models.vllm_patch.recording.store import (
    MemoryStore,
    RecordStore,
    SqlStore,
)

#: Attribute on app.state holding the active RecordStore.
_STORE_STATE_ATTR = "trinity_record_store"
#: Attribute on app.state holding the active Recorder.
_RECORDER_STATE_ATTR = "trinity_recorder"


def setup_server_in_ray(args, logger):
    """Validate API server args, set up signal handler, create socket
    ready to serve.

    Copied verbatim from api_patch_v17.py — identical lifecycle so the
    recording entry point behaves like the stock Trinity server.
    """

    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock


def dummy_add_signal_handler(self, *args, **kwargs):
    # DO NOTHING HERE
    pass


def _setup_recording(
    args,
    engine_client,
    app,
    logger,
    recording_config: Optional[RecordingConfig] = None,
) -> Optional[Recorder]:
    """Wire generation recording onto the in-construction server.

    Returns the started Recorder (for lifecycle management), or None if
    recording is disabled (``recording_config`` is None).

    The static config (db_url/table/topk) arrives explicitly via
    ``recording_config`` (built by ``get_api_server`` from
    ``InferenceModelConfig``). The *dynamic* checkpoint version is read live
    off ``engine_client.trinity_model_version`` (mirrored by VLLMModel at
    engine creation and in ``sync_model_weights``).

    Args:
        args: Parsed vLLM CLI args.
        engine_client: AsyncLLM instance (we own it pre-init_app_state).
        app: FastAPI app from ``build_app`` (we own it pre-serve_http).
        logger: Logger.
        recording_config: Static recording config; None disables recording.
    """
    if recording_config is None:
        return None

    if recording_config.db_url:
        store: RecordStore = SqlStore(
            db_url=recording_config.db_url, table_name=recording_config.table
        )
    else:
        logger.warning(
            "recording enabled but recording_config.db_url is None; falling "
            "back to in-process MemoryStore (no cross-process visibility)"
        )
        store = MemoryStore()

    # Rank is constant per process; capture once (RequestOutput does not expose
    # parallel_config, so we read it from engine_client here, mirroring
    # api_patch_v17.py:148).
    try:
        rank = int(engine_client.vllm_config.parallel_config._api_process_rank)
    except Exception:
        rank = 0

    recorder = Recorder(
        store=store,
        topk=recording_config.topk,
        enabled=True,
        rank=rank,
        engine_client=engine_client,
    )

    # (1) engine-level wrap — before init_app_state so serving objects inherit
    #     the wrapped reference. Idempotent via the __patched_*__ guard.
    patch_engine_for_recording(engine_client, recorder, logger)

    # (2) in-process middleware: API key -> contextvar. Zero network hop.
    app.add_middleware(RecordingIdentityMiddleware)

    # (3) query routes mounted on the main app; OpenAI /v1/* surface untouched.
    app.include_router(query_router)

    setattr(app.state, _STORE_STATE_ATTR, store)
    setattr(app.state, _RECORDER_STATE_ATTR, recorder)

    logger.info(
        "Generation recording enabled: topk=%d store=%s rank=%d",
        recording_config.topk,
        type(store).__name__,
        rank,
    )
    return recorder


async def run_server_worker_in_ray(
    listen_address,
    sock,
    args,
    engine_client,
    logger,
    recording_config: Optional[RecordingConfig] = None,
) -> None:
    """Modified from vllm.entrypoints.openai.api_server.run_server_worker.

    Differs from api_patch_v17.py only in the recording wiring inserted between
    ``build_app`` and ``init_app_state``, plus starting/stopping the recorder
    flusher around ``serve_http``.
    """
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    app = build_app(args)

    # --- recording wiring: engine wrap must precede init_app_state -----------
    recorder = _setup_recording(args, engine_client, app, logger, recording_config=recording_config)
    # ------------------------------------------------------------------------

    await init_app_state(engine_client, app.state, args)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler = functools.partial(dummy_add_signal_handler, loop)

    logger.info(
        "Starting vLLM API server %d on %s",
        engine_client.vllm_config.parallel_config._api_process_rank,
        listen_address,
    )

    if recorder is not None:
        recorder.start()

    shutdown_task = await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
    )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        if recorder is not None:
            await recorder.stop()
        sock.close()


async def run_server_in_ray(
    args,
    engine_client,
    logger,
    recording_config: Optional[RecordingConfig] = None,
):
    # Modified from vllm.entrypoints.openai.api_server.run_server
    listen_address, sock = setup_server_in_ray(args, logger)
    logger.info("vLLM API server listening on %s", listen_address)
    await run_server_worker_in_ray(
        listen_address, sock, args, engine_client, logger, recording_config
    )


async def run_api_server_with_recording(
    async_llm,
    host: str,
    port: int,
    model_path: str,
    logger: logging.Logger,
    chat_template: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    enable_log_requests: bool = False,
    recording_config: Optional[RecordingConfig] = None,
):
    """Drop-in recording-enabled variant of
    ``api_patch_v17.run_api_server_in_ray_actor_v17``.

    Same signature plus an optional ``recording_config`` so launchers can
    switch by import path. Requires vllm >= 0.17.0. Recording is on iff
    ``recording_config`` is provided (built by ``get_api_server`` from
    ``InferenceModelConfig`` when ``enable_recording`` is on). The dynamic
    checkpoint version is read off ``async_llm.trinity_model_version``
    (mirrored by VLLMModel), so it is not part of the static config here.
    """
    vllm_version = get_vllm_version()
    if vllm_version < parse_version("0.17.0"):
        raise ValueError(
            f"Unsupported vllm version: {vllm.__version__}. "
            "This patch requires vllm version >= 0.17.0"
        )

    parser = FlexibleArgumentParser(description="Run the OpenAI API server.")
    args = make_arg_parser(parser)
    cli_args = [
        "--host",
        str(host),
        "--port",
        str(port),
        "--model",
        model_path,
        "--enable-server-load-tracking",  # enable tracking for load balancing
    ]
    if enable_log_requests:
        cli_args.append("--enable-log-requests")
    if enable_auto_tool_choice:
        cli_args.append("--enable-auto-tool-choice")
    if tool_call_parser:
        cli_args.extend(["--tool-call-parser", tool_call_parser])
    if reasoning_parser:
        cli_args.extend(["--reasoning-parser", reasoning_parser])
    if chat_template:
        cli_args.extend(["--chat-template", chat_template])

    # NOTE: routed_experts capture and the logprobs cap are ENGINE-level
    # ModelConfig fields (consumed by the scheduler/worker, not the API serving
    # layer), so they take effect at engine build time — which in this launch
    # path happens in VLLMModel (via EngineArgs), *before* this runner gets the
    # already-built ``async_llm``. Adding ``--enable-return-routed-experts`` /
    # ``--max-logprobs`` here would be inert (init_app_state does not read them).
    # The Allocator therefore forces ``InferenceModelConfig.enable_return_routed_experts
    # = True`` when recording is on, and the engine's default ``max_logprobs=20``
    # covers the recorder's top-k (``VLLM_RECORD_TOPK``, default 1). To record
    # routed_experts, the engine must be built with that flag on — the launcher
    # is responsible for that, not these CLI args.

    args = parser.parse_args(cli_args)
    args.structured_outputs_config.reasoning_parser = reasoning_parser
    logger.info(f"Starting vLLM OpenAI API server with args: {args}")
    await run_server_in_ray(args, async_llm, logger, recording_config)
