# -*- coding: utf-8 -*-
"""Trinity-specific worker extending veRL's engine-based ActorRolloutRefWorker.

This replaces the old fsdp_workers.py and megatron_workers.py with a single
thin extension that adds Trinity-specific hooks on top of veRL's unified
engine-based training worker.
"""
from typing import Optional

from omegaconf import DictConfig
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from trinity.common.config import AlgorithmConfig
from trinity.trainer.verl.losses import build_trinity_loss


class TrinityActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extends veRL's ActorRolloutRefWorker with Trinity-specific hooks.

    Additions over the base class:
    - set_algorithm(): injects Trinity's pluggable loss function (policy + KL + entropy)
    - Applies Trinity-specific monkey patches after model init
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config=config, role=role, **kwargs)
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
