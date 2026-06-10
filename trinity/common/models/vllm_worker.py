# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import logging

import ray
import torch
import torch.distributed

from trinity.common.models.vllm_patch.worker_patch import patch_vllm_prompt_logprobs
from trinity.common.weight_transfer import ModelWeightReceiver
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.distributed import init_process_group
from trinity.utils.log import get_logger


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

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        explorer_name: str = None,
        namespace: str = None,
        zmq_ip: str = None,
        zmq_port: int = None,
        bucket_size_mb: int = 500,
    ):
        """Init torch process group for model weights update"""
        rank = torch.distributed.get_rank()
        self.logger = get_logger(f"vllm_worker_{rank}")

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        self._state_dict_meta = None
        self._weight_update_rank = rank + rank_offset
        self.logger.info(
            f"vLLM starting init_process_group:\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank={rank}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}"
        )
        self._model_update_group = init_process_group(
            host=master_address,
            port=master_port,
            group_name=group_name,
            backend=backend,
            timeout=timeout,
            world_size=world_size,
            rank=self._weight_update_rank,
        )
        self.logger.info("vLLM init_process_group finished.")
        self._explorer_name = explorer_name
        self._namespace = namespace
        self.synchronizer = Synchronizer.get_actor(namespace=self._namespace)
        self._checkpoint_converter = None

        # Set up the bucketed weight receiver for NCCL transfer.
        self._weight_receiver = None
        if zmq_ip is not None and zmq_port is not None:
            bucket_size = bucket_size_mb * 1024 * 1024
            self._weight_receiver = ModelWeightReceiver(
                bucket_size=bucket_size,
            )
            self._weight_receiver.prepare()
            self._weight_receiver.init_process_group(self._model_update_group)
            self._weight_receiver.connect_metadata(zmq_ip, zmq_port)
            self.logger.info(
                f"ModelWeightReceiver ready "
                f"(ZMQ: {zmq_ip}:{zmq_port}, bucket_size={bucket_size_mb}MB)"
            )

    def set_state_dict_meta(self, state_dict_meta: list):
        """Set the state_dict meta for NCCL weight sync."""
        self._state_dict_meta = state_dict_meta

    def teardown_process_group(self):
        """Destroy the NCCL process group and finalize the receiver."""
        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            self._weight_receiver.finalize()
            self._weight_receiver = None
        if hasattr(self, "_model_update_group") and self._model_update_group is not None:
            torch.distributed.destroy_process_group(self._model_update_group)
            self._model_update_group = None

    def update_weight(self):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if self._weight_update_rank == 0:
            state_dict, model_version = ray.get(self.synchronizer.get_model_state_dict.remote())
            if isinstance(state_dict, tuple):
                # currently only megatron return a tuple
                method, checkpoint_dir = state_dict
                if method == "megatron":
                    if self._checkpoint_converter is None:
                        from trinity.common.models.utils import get_megatron_converter

                        self._checkpoint_converter = get_megatron_converter(checkpoint_dir)
                    state_dict = self._checkpoint_converter.get_state_dict(checkpoint_dir)
                elif method == "huggingface":
                    from trinity.common.models.utils import load_huggingface_state_dict

                    state_dict = load_huggingface_state_dict(checkpoint_dir)
                else:
                    raise NotImplementedError(f"{method} is not supported")
                ray.get(self.synchronizer.set_model_state_dict.remote(state_dict, model_version))
        if self._state_dict_meta is None:
            self._state_dict_meta = ray.get(self.synchronizer.get_state_dict_meta.remote())

        def _weight_iterator():
            for name, dtype_str, shape in self._state_dict_meta:
                if self._weight_update_rank == 0:
                    weight = state_dict[name]
                    weight = weight.to(self.device)
                else:
                    dtype = getattr(torch, dtype_str)
                    weight = torch.empty(shape, dtype=dtype, device=self.device)
                torch.distributed.broadcast(weight, 0, group=self._model_update_group)
                yield (name, weight)

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.model_runner.vllm_config):
            self.model_runner.reload_weights(
                weights_iterator=_weight_iterator(),
                is_checkpoint_format=True,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def update_weight_nccl(self):
        """Receive weights via bucketed NCCL broadcast (ModelWeightReceiver).

        Uses ``receive_sync()`` because vLLM's ``reload_weights`` expects a
        synchronous iterator.  Falls back to the legacy ``update_weight()``
        if the receiver was not set up (e.g. ZMQ metadata was not provided).
        """
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            return self.update_weight()

        weights_iter = self._weight_receiver.receive_sync()

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.model_runner.vllm_config):
            self.model_runner.reload_weights(
                weights_iterator=weights_iter,
                is_checkpoint_format=True,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
