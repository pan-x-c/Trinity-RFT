# -*- coding: utf-8 -*-
"""Trinity-specific worker extending veRL's engine-based ActorRolloutRefWorker.

This replaces the old fsdp_workers.py and megatron_workers.py with a single
thin extension that adds Trinity-specific hooks on top of veRL's unified
engine-based training worker.
"""
from typing import Optional

import ray
import torch
from omegaconf import DictConfig
from verl.single_controller.base.decorator import Dispatch, register
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
    - Applies Trinity-specific monkey patches after model init
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config=config, role=role, **kwargs)
        self.logger = get_logger(f"{role}_{self.rank}", in_ray_actor=True)
        self._is_rollout = False  # Disable rollout in Trainer
        self._algo_config: Optional[AlgorithmConfig] = None

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

        super().init_model()

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

        This is called by VerlPPOTrainerWrapper after worker initialization to
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

        This method sets up the worker group responsible for synchronizing
        model weights across multiple workers during training. It ensures that
        all workers in the group have the same model parameters before training
        begins.
        """
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        if torch.distributed.get_rank() == 0:
            self.state_dict_meta_list = [
                (name, str(param.dtype).split(".")[-1], param.shape)
                for name, param in per_tensor_param
            ]
            self.get_().empty_cache()
            master_address, master_port = self.get_available_master_addr_port()
            world_size = self.config.synchronizer.explorer_world_size + 1
            self.logger.info(
                f"Trainer init_process_group {master_address}:{master_port} ({world_size})."
            )
            synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)
            setup_ref = synchronizer.setup_weight_sync_group.remote(
                master_address, master_port, self.state_dict_meta_list
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

        Overrides the base class method to ensure that after syncing weights, we
        also sync the optimizer states if needed (e.g., for Adam optimizers).
        """
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        for _, param in per_tensor_param:
            torch.distributed.broadcast(param, src=0, group=self._model_update_group)
