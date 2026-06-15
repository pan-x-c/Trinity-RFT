# -*- coding: utf-8 -*-
"""Extensions for vLLM."""
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    pack_tensors,
    unpack_tensor,
)

from trinity.common.models.vllm_patch.worker_patch import patch_vllm_prompt_logprobs


def _suppress_layerwise_reload_warnings() -> None:
    """Silence benign vLLM layerwise reload warnings during weight sync."""
    try:
        logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        if logger is not None:
            logger.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - best-effort suppression
        pass


class WorkerExtension:
    def apply_patches(self):
        """Apply necessary patches to vLLM."""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        patch_vllm_prompt_logprobs(self.model_runner)
        _suppress_layerwise_reload_warnings()


def load_state_dict(checkpoint_path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Load checkpoint tensors as ``(name, tensor)`` pairs.

    Replace this placeholder with your checkpoint reader. Rank 0 is the only
    rank that calls this function; other ranks receive tensors over NCCL.
    """
    raise NotImplementedError(
        "Provide a load_state_dict(checkpoint_path) implementation for " f"{checkpoint_path!r}."
    )


@dataclass
class CheckpointWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the checkpoint weight transfer example."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


@dataclass
class CheckpointWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for loading a checkpoint from disk."""

    checkpoint_path: str
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS


@dataclass
class _PackedChunkMetadata:
    names: list[str]
    shapes: list[list[int]]
    dtype_names: list[str]
    tensor_sizes: list[int]


class CheckpointWeightTransferEngine(
    WeightTransferEngine[CheckpointWeightTransferInitInfo, CheckpointWeightTransferUpdateInfo]
):
    """Example engine that reads checkpoints on rank 0 and broadcasts weights.

    This is a teaching example, not a built-in backend. Register it at runtime
    with ``WeightTransferEngineFactory.register_engine`` before creating the LLM.
    """

    init_info_cls = CheckpointWeightTransferInitInfo
    update_info_cls = CheckpointWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, parallel_config, model)
        self.model_update_group: PyNcclCommunicator | None = None

    def init_transfer_engine(self, init_info: CheckpointWeightTransferInitInfo) -> None:
        dp_rank = self.parallel_config.data_parallel_index
        world_size_per_dp = self.parallel_config.world_size
        rank_within_dp = self.parallel_config.rank

        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset
        device = torch.accelerator.current_device_index()
        self.model_update_group = NCCLWeightTransferEngine._stateless_init_process_group(
            init_info.master_address,
            init_info.master_port,
            rank,
            init_info.world_size,
            device=device,
        )

    def receive_weights(
        self,
        update_info: CheckpointWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self.model_update_group is None:
            raise RuntimeError(
                "Checkpoint weight transfer not initialized. " "Call init_transfer_engine() first."
            )

        if self.model_update_group.rank == 0:
            self._produce_checkpoint_weights(update_info, load_weights)
        else:
            self._consume_checkpoint_weights(load_weights)

    def _produce_checkpoint_weights(
        self,
        update_info: CheckpointWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        assert self.model_update_group is not None

        iterator = load_state_dict(update_info.checkpoint_path)

        def post_iter_func(item: tuple[str, torch.Tensor]) -> torch.Tensor:
            tensor = item[1]
            if tensor.device.type == "cuda":
                return tensor
            return tensor.to(device="cuda", non_blocking=True)

        while True:
            chunk = pack_tensors(
                iterator=iterator,
                post_iter_func=post_iter_func,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
            )
            if chunk is None:
                self.model_update_group.group.broadcast_obj(None, src=0)
                break

            metadata = _PackedChunkMetadata(
                names=chunk.names,
                shapes=chunk.shapes,
                dtype_names=[str(dtype).split(".")[-1] for dtype in chunk.dtypes],
                tensor_sizes=chunk.tensor_sizes,
            )
            self.model_update_group.group.broadcast_obj(metadata, src=0)
            self.model_update_group.broadcast(
                chunk.packed_tensor, src=0, stream=torch.cuda.current_stream()
            )
            load_weights(
                unpack_tensor(
                    chunk.packed_tensor,
                    metadata.names,
                    metadata.shapes,
                    chunk.dtypes,
                    metadata.tensor_sizes,
                )
            )

    def _consume_checkpoint_weights(
        self,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        assert self.model_update_group is not None

        while True:
            metadata = self.model_update_group.group.broadcast_obj(None, src=0)
            if metadata is None:
                break
            assert isinstance(metadata, _PackedChunkMetadata)

            dtypes = [getattr(torch, dtype_name) for dtype_name in metadata.dtype_names]
            packed_tensor = torch.empty(
                sum(metadata.tensor_sizes), dtype=torch.uint8, device="cuda"
            )
            self.model_update_group.broadcast(
                packed_tensor, src=0, stream=torch.cuda.current_stream()
            )
            load_weights(
                unpack_tensor(
                    packed_tensor,
                    metadata.names,
                    metadata.shapes,
                    dtypes,
                    metadata.tensor_sizes,
                )
            )

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "CheckpointWeightTransferEngine reads weights from disk on "
            "inference rank 0. Call update_weights with a checkpoint_path "
            "instead of trainer_send_weights."
        )
