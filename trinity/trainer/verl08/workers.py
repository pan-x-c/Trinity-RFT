# -*- coding: utf-8 -*-
"""Trinity-specific worker extending veRL's engine-based ActorRolloutRefWorker.

This replaces the old fsdp_workers.py and megatron_workers.py with a single
thin extension that adds Trinity-specific hooks on top of veRL's unified
engine-based training worker.

Key additions over the base class:
- set_algorithm(): injects Trinity's pluggable loss function
- save_state_dict / upload_state_dict / sync_weight_nccl: Trinity-specific
  checkpoint and weight sync methods that delegate to strategy-specific helpers
- init_weights_update_group: NCCL weight sync group setup for Explorer↔Trainer
"""
from typing import Optional

import ray
import torch
from omegaconf import DictConfig
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers import ActorRolloutRefWorker

from trinity.common.config import AlgorithmConfig
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl08.losses import build_trinity_loss
from trinity.utils.distributed import init_process_group
from trinity.utils.log import get_logger


class TrinityActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extends veRL's ActorRolloutRefWorker with Trinity-specific hooks.

    Additions over the base class:
    - set_algorithm(): injects Trinity's pluggable loss function (policy + KL + entropy)
    - save_state_dict / upload_state_dict: checkpoint and memory-based weight sync
    - sync_weight_nccl: NCCL-based weight broadcast for Explorer↔Trainer
    - init_weights_update_group: sets up NCCL process group for weight sync
    - Applies Trinity-specific monkey patches after model init
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config=config, role=role, **kwargs)
        self.logger = get_logger(f"{role}_{self.rank}", in_ray_actor=True)
        self._is_rollout = False  # Disable rollout in Trainer
        self._algo_config: Optional[AlgorithmConfig] = None
        self._model_update_group = None
        self._state_dict_meta_list = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model, then apply Trinity-specific monkey patches.

        veRL main already applies its own monkey patches (ulysses SP, prefix
        grouper, tiled MLP, VLM patching) inside the engine. Trinity's
        monkey_patch module adds patches that veRL main does NOT have:
        - Qwen3.5 GatedDeltaNet / hybrid-attention layer support
        - GLM4V text model forward
        - Fused kernels VLM SP bugfix (patch_fused_kernels)
        - Flops counter registration for qwen3_5
        """
        from trinity.trainer.verl.monkey_patch import apply_monkey_patch

        # Strip "rollout" from role so the base class skips rollout engine init.
        # veRL checks `if "rollout" in self.role:` to decide whether to build
        # the rollout engine — Trinity handles rollout in Explorer, not Trainer.
        original_role = self.role
        self.role = self.role.replace("_rollout", "")
        super().init_model()
        self.role = original_role

        # Apply Trinity-specific patches on top of what veRL already did
        if self.actor is not None and hasattr(self.actor, "engine"):
            model = getattr(self.actor.engine, "model", None)
            if model is not None:
                ulysses_sp_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
                use_remove_padding = self.config.model.get("use_remove_padding", False)
                use_fused_kernels = self.config.model.get("use_fused_kernels", False)
                fused_kernel_options = self.config.model.get("fused_kernel_options", None)
                fused_kernels_backend = (
                    fused_kernel_options.get("impl_backend", None)
                    if fused_kernel_options is not None
                    else None
                )
                apply_monkey_patch(
                    model=model,
                    ulysses_sp_size=ulysses_sp_size,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    fused_kernels_backend=fused_kernels_backend,
                )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_algorithm(self, algo_config: AlgorithmConfig):
        """Set Trinity's algorithm config and rebuild loss function.

        This is called by VERLTrainer after worker initialization to
        inject the pluggable policy loss, KL loss, and entropy loss from
        Trinity's algorithm registry.
        """
        self._algo_config = algo_config
        if self.actor is not None:
            loss_fn = build_trinity_loss(algo_config)
            self.actor.set_loss_fn(loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_weights_update_group(self):
        """Initialize the weight update group for distributed training.

        This method sets up the NCCL process group responsible for synchronizing
        model weights between the Trainer and Explorer. It runs only on rank 0
        and coordinates with the Synchronizer actor.
        """
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        if torch.distributed.get_rank() == 0:
            self._state_dict_meta_list = [
                (name, str(param.dtype).split(".")[-1], param.shape)
                for name, param in per_tensor_param
            ]
            aggressive_empty_cache(force_sync=True)
            master_address, master_port = self.get_available_master_addr_port()
            world_size = self.config.synchronizer.explorer_world_size + 1
            self.logger.info(
                f"Trainer init_process_group {master_address}:{master_port} ({world_size})."
            )
            synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)
            setup_ref = synchronizer.setup_weight_sync_group.remote(
                master_address, master_port, self._state_dict_meta_list
            )
            timeout = self.config.synchronizer.sync_timeout

            self.logger.info("Trainer start init_process_group.")
            self._model_update_group = init_process_group(
                host=master_address,
                port=int(master_port),
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                backend="nccl",
                timeout=timeout,
                world_size=world_size,
                rank=0,
            )
            self.logger.info("Trainer init_process_group done, wait for explorer confirmation.")
            ray.get(setup_ref)
            self.logger.info("Trainer explorer setup confirmation received.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_weight_nccl(self):
        """Sync model weights across workers using NCCL.

        Broadcasts full model parameters from rank 0 (trainer) to all
        Explorer ranks via the NCCL process group.
        """
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        for _, param in per_tensor_param:
            torch.distributed.broadcast(param, src=0, group=self._model_update_group)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_state_dict(self, local_path, global_step=0):
        """Save model state dict for checkpoint-based weight sync.

        Delegates to strategy-specific helpers in fsdp_engine.py or
        megatron_engine.py based on the actor's strategy.
        """
        strategy = self.config.actor.strategy
        if strategy.startswith("fsdp"):
            from trinity.trainer.verl08.fsdp_engine import fsdp_save_state_dict

            fsdp_save_state_dict(self.actor.engine, local_path, global_step)
        elif strategy.startswith("megatron"):
            from trinity.trainer.verl08.megatron_engine import megatron_save_state_dict

            megatron_save_state_dict(self.actor.engine, local_path, global_step)
        else:
            raise ValueError(f"Unsupported strategy for save_state_dict: {strategy}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def upload_state_dict(self, global_step=0):
        """Upload model state dict to Synchronizer for memory-based weight sync.

        Delegates to strategy-specific helpers in fsdp_engine.py or
        megatron_engine.py based on the actor's strategy.
        """
        strategy = self.config.actor.strategy
        synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)

        if strategy.startswith("fsdp"):
            from trinity.trainer.verl08.fsdp_engine import fsdp_upload_state_dict

            fsdp_upload_state_dict(self.actor.engine, synchronizer, global_step)
        elif strategy.startswith("megatron"):
            from trinity.trainer.verl08.megatron_engine import (
                megatron_upload_state_dict,
            )

            megatron_upload_state_dict(self.actor.engine, synchronizer, global_step)
        else:
            raise ValueError(f"Unsupported strategy for upload_state_dict: {strategy}")
