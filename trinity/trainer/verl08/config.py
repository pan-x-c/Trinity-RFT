# -*- coding: utf-8 -*-
"""veRL 0.8 configuration builder for Trinity-RFT.

This module provides `build_verl_config()`, the single entry point that
converts Trinity's `Config` (global_config) into the `DictConfig` required
by veRL 0.8's `ActorRolloutRefWorker` and `TrainingWorker`.

Design principle (P1):
  - VERLTrainer uses `global_config` for all Trinity-level logic.
  - `build_verl_config()` is called **once** to produce the minimal `DictConfig`
    needed at the Worker/Engine boundary.
  - Only fields that veRL workers/engines actually consume are included.

The DictConfig structure must match what `ActorRolloutRefWorker.__init__`
and `ActorRolloutRefWorker.init_model()` expect:

  config.model        → HFModelConfig       (via omega_conf_to_dataclass)
  config.actor        → FSDPActorConfig | McoreActorConfig
  config.ref          → FSDPActorConfig | McoreActorConfig  (subset)
  config.rollout      → RolloutConfig
  config.synchronizer → SynchronizerConfig   (direct access)
  config.trainer      → trainer-level fields (gpu_per_node, node_num, etc.)
  config.critic       → CriticConfig         (for VERLTrainer._init_workers)
  config.global_profiler → profiler settings
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.config import Config
from trinity.common.config import OptimizerConfig as TrinityOptimizerConfig
from trinity.common.config import set_if_none
from trinity.common.constants import EXPLORER_NAME
from trinity.utils.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight dataclasses for trainer-level config (stored in DictConfig.trainer)
# These are NOT veRL config classes — they are Trinity-specific containers
# that VERLTrainer reads from self.config.trainer.
# ---------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    """Profiler configuration for veRL worker groups."""

    steps: Optional[int] = None
    tools: str = "nsys"


@dataclass
class VerlTrainerConfig:
    """Trainer-level config stored in the DictConfig.

    These fields are read by VERLTrainer, NOT by veRL workers directly.
    """

    gpu_per_node: int = 8
    node_num: int = 1
    total_training_steps: int = -1
    sync_freq: int = 0
    save_freq: int = 0
    default_local_dir: str = ""
    default_hdfs_dir: Optional[str] = None
    max_actor_ckpt_to_keep: Optional[int] = None
    max_critic_ckpt_to_keep: Optional[int] = None
    del_local_ckpt_after_load: bool = False
    resume_mode: str = "auto"


def build_verl_config(global_config: Config) -> DictConfig:  # noqa: C901
    """Build the veRL 0.8 DictConfig from Trinity's global Config.

    This produces the *minimal* DictConfig that ActorRolloutRefWorker and
    TrainingWorker need.  All Trinity-level logic (algorithm, advantage,
    KL penalty, etc.) stays in VERLTrainer using global_config.

    The resulting DictConfig has the following top-level keys:
      - model:       HFModelConfig fields
      - actor:       FSDPActorConfig or McoreActorConfig fields
      - ref:         actor-style config for reference model
      - rollout:     RolloutConfig fields
      - critic:      CriticConfig fields (for VERLTrainer._init_workers)
      - synchronizer: SynchronizerConfig
      - trainer:     VerlTrainerConfig fields
      - global_profiler: ProfilerConfig fields
    """
    cfg = global_config
    strategy = cfg.trainer.trainer_strategy  # "fsdp", "fsdp2", or "megatron"
    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    total_training_steps = cfg.trainer.total_steps or sys.maxsize
    algorithm = ALGORITHM_TYPE.get(cfg.algorithm.algorithm_type)
    use_critic = algorithm.use_critic

    # ====================================================================
    # 1. Model config (HFModelConfig fields)
    # ====================================================================
    model = _build_model_config(cfg)

    # ====================================================================
    # 2. Actor config
    # ====================================================================
    actor = _build_actor_config(cfg, strategy, total_training_steps)

    # ====================================================================
    # 3. Ref config
    # ====================================================================
    ref = _build_ref_config(cfg, strategy)

    # ====================================================================
    # 4. Rollout config
    # ====================================================================
    rollout = _build_rollout_config(cfg, actor)

    # ====================================================================
    # 5. Critic config
    # ====================================================================
    critic = _build_critic_config(cfg, strategy, use_critic, total_training_steps)

    # ====================================================================
    # 6. Trainer config (Trinity-specific, read by VERLTrainer)
    # ====================================================================
    trainer = _build_trainer_config(cfg, total_training_steps)

    # ====================================================================
    # 7. Global profiler
    # ====================================================================
    global_profiler = {"steps": None, "tools": "nsys"}

    # ====================================================================
    # 8. Assemble the DictConfig
    # ====================================================================
    verl_dict = {
        "model": model,
        "actor": actor,
        "ref": ref,
        "rollout": rollout,
        "critic": critic,
        "synchronizer": cfg.synchronizer,
        "trainer": trainer,
        "global_profiler": global_profiler,
    }

    return OmegaConf.create(verl_dict, flags={"allow_objects": True})


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_model_config(cfg: Config) -> dict:
    """Build HFModelConfig-compatible dict from Trinity config."""
    model = {
        "path": cfg.model.model_path,
        "use_shm": False,
        "trust_remote_code": cfg.model.trust_remote_code,
        # LoRA fields (set later if lora_configs is not None)
        "lora_rank": 0,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "exclude_modules": None,
        "lora_adapter_path": None,
        # Other HFModelConfig fields
        "enable_gradient_checkpointing": True,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
        "fused_kernel_options": {},
        "custom_chat_template": cfg.model.custom_chat_template,
        "enable_activation_offload": False,
        "external_lib": None,
        "override_config": {},
        "mtp": {"enable": False},
    }

    # Apply LoRA config if present
    if cfg.model.lora_configs is not None:
        lora_config = cfg.model.lora_configs[0]
        model["lora_rank"] = lora_config.lora_rank
        model["lora_alpha"] = lora_config.lora_alpha
        model["target_modules"] = lora_config.target_modules
        model["exclude_modules"] = lora_config.exclude_modules
        if not lora_config.is_dummy:
            model["lora_adapter_path"] = lora_config.path

    # Rope config
    if cfg.model.rope_scaling is not None:
        model["override_config"]["rope_scaling"] = cfg.model.rope_scaling
    if cfg.model.rope_theta is not None:
        model["override_config"]["rope_theta"] = cfg.model.rope_theta

    return model


def _build_actor_config(cfg: Config, strategy: str, total_training_steps: int) -> dict:
    """Build ActorConfig-compatible dict from Trinity config."""
    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    actor = {
        "strategy": strategy,
        "ppo_mini_batch_size": cfg.buffer.train_batch_size,
        "ppo_micro_batch_size_per_gpu": None,
        "ppo_micro_batch_size": None,
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_infer_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "entropy_coeff": 0,
        "use_kl_loss": cfg.algorithm.kl_loss_fn != "none",
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "ppo_epochs": 1,
        "shuffle": False,
        "data_loader_seed": 42,
        "grad_clip": cfg.trainer.grad_clip,
        "clip_grad": cfg.trainer.grad_clip,
        "loss_agg_mode": cfg.algorithm.loss_agg_mode or "token-mean",
        "loss_scale_factor": None,
        "calculate_sum_pi_squared": False,
        "use_prefix_grouper": False,
        "use_torch_compile": True,
        "freeze_vision_tower": False,
        "use_fused_kernels": False,
        "rollout_n": cfg.algorithm.repeat_times,
        "fix_actor_microbatch_loss_scale": cfg.trainer.fix_actor_microbatch_loss_scale,
        "ulysses_sequence_parallel_size": cfg.trainer.ulysses_sequence_parallel_size,
        "router_replay": {"mode": "disabled"},
        "policy_loss": {
            "loss_mode": "vanilla",
            "rollout_correction": cfg.algorithm.rollout_correction or {"bypass_mode": True},
        },
        "profiler": {},
    }

    # Strategy-specific engine config
    if is_fsdp:
        actor["fsdp_config"] = _build_fsdp_engine_config(cfg, strategy)
        actor["entropy_from_logits_with_chunking"] = False
        actor["entropy_checkpointing"] = False
        actor["use_remove_padding"] = cfg.trainer.use_remove_padding
        actor["use_rollout_log_probs"] = False
    elif is_megatron:
        actor["megatron"] = _build_mcore_engine_config(cfg)
        actor["load_weight"] = True
        actor["use_rollout_log_probs"] = False

    # Optimizer config
    actor["optim"] = _build_optimizer_config(
        cfg.algorithm.optimizer, strategy, total_training_steps
    )

    # Checkpoint config
    actor["checkpoint"] = _build_checkpoint_config()

    return actor


def _build_ref_config(cfg: Config, strategy: str) -> dict:
    """Build ref-config dict (subset of actor config)."""
    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    ref = {
        "strategy": strategy,
        "log_prob_micro_batch_size_per_gpu": None,
        "log_prob_use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "log_prob_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ulysses_sequence_parallel_size": cfg.trainer.ulysses_sequence_parallel_size,
        "use_prefix_grouper": False,
        "entropy_from_logits_with_chunking": False,
        "entropy_checkpointing": False,
        "load_weight": True,
        "profiler": {},
        "router_replay": {"mode": "disabled"},
    }

    if is_fsdp:
        ref["fsdp_config"] = _build_fsdp_engine_config(cfg, strategy)
        ref["use_remove_padding"] = cfg.trainer.use_remove_padding
        ref["use_rollout_log_probs"] = False
    elif is_megatron:
        ref["megatron"] = _build_mcore_engine_config(cfg)
        ref["load_weight"] = True
        ref["use_rollout_log_probs"] = False

    ref["checkpoint"] = _build_checkpoint_config(save_contents=["model"], load_contents=["model"])

    return ref


def _build_rollout_config(cfg: Config, actor: dict) -> dict:
    """Build RolloutConfig-compatible dict."""
    # Get temperature from taskset or default
    temperature = 1.0
    if cfg.buffer.explorer_input.tasksets:
        temperature = cfg.buffer.explorer_input.tasksets[0].rollout_args.temperature

    rollout = {
        "name": "auto",
        "mode": "async",
        "temperature": temperature,
        "n": cfg.algorithm.repeat_times,
        # log prob settings mirror actor settings
        "log_prob_micro_batch_size": None,
        "log_prob_micro_batch_size_per_gpu": None,
        "log_prob_use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "log_prob_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        # Multi-turn / val (Trinity doesn't use these)
        "val_kwargs": {"do_sample": False},
        "multi_turn": {"enable": False},
        "checkpoint_engine": {
            "backend": "naive",
            "update_weights_bucket_megabytes": 2048,
            "engine_kwargs": {},
            "custom_backend_module": None,
        },
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "enable_sleep_mode": True,
    }

    return rollout


def _build_critic_config(
    cfg: Config, strategy: str, use_critic: bool, total_training_steps: int
) -> dict:
    """Build CriticConfig-compatible dict for VERLTrainer._init_workers."""
    critic = {
        "enable": use_critic,
        "strategy": strategy,
        "ppo_mini_batch_size": cfg.buffer.train_batch_size,
        "ppo_micro_batch_size_per_gpu": None,
        "ppo_micro_batch_size": None,
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_infer_max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "ppo_epochs": 1,
        "shuffle": True,
        "cliprange_value": 0.5,
        "loss_agg_mode": "token-mean",
        "data_loader_seed": 42,
        "rollout_n": cfg.algorithm.repeat_times,
        "grad_clip": cfg.trainer.grad_clip,
        "ulysses_sequence_parallel_size": cfg.trainer.ulysses_sequence_parallel_size,
    }

    # Critic model config
    critic_model_path = cfg.model.critic_model_path or cfg.model.model_path
    critic["model_config"] = {
        "path": critic_model_path,
        "model_type": "value_model",
        "use_shm": False,
        "trust_remote_code": cfg.model.trust_remote_code,
        "enable_gradient_checkpointing": True,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "override_config": {},
    }

    # Rope config for critic
    if cfg.model.rope_scaling is not None:
        critic["model_config"]["override_config"]["rope_scaling"] = cfg.model.rope_scaling
    if cfg.model.rope_theta is not None:
        critic["model_config"]["override_config"]["rope_theta"] = cfg.model.rope_theta

    # Strategy-specific engine config
    is_fsdp = strategy.startswith("fsdp")
    is_megatron = strategy.startswith("megatron")

    if is_fsdp:
        critic["fsdp"] = _build_fsdp_engine_config(cfg, strategy)
        critic["forward_micro_batch_size"] = 1
        critic["forward_micro_batch_size_per_gpu"] = 1
        critic["forward_max_token_len_per_gpu"] = cfg.trainer.max_token_len_per_gpu
    elif is_megatron:
        critic["megatron"] = _build_mcore_engine_config(cfg)
        critic["load_weight"] = True

    # Critic optimizer — use a default conservative config
    critic["optim"] = _build_critic_optimizer_config(strategy, total_training_steps)
    critic["checkpoint"] = _build_checkpoint_config()
    critic["profiler"] = {}

    return critic


def _build_fsdp_engine_config(cfg: Config, strategy: str) -> dict:
    """Build FSDPEngineConfig-compatible dict."""
    fsdp = {
        "param_offload": False,
        "optimizer_offload": False,
        "offload_policy": False,
        "reshard_after_forward": True,
        "wrap_policy": {"min_num_params": 0},
        "fsdp_size": -1,
        "forward_prefetch": False,
        "model_dtype": None,
        "dtype": "bfloat16",
        "mixed_precision": {},
        "ulysses_sequence_parallel_size": cfg.trainer.ulysses_sequence_parallel_size,
        "strategy": strategy,  # "fsdp" or "fsdp2"
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
    }
    return fsdp


def _build_mcore_engine_config(cfg: Config) -> dict:
    """Build McoreEngineConfig-compatible dict."""
    megatron = {
        "strategy": "megatron",
        "param_offload": False,
        "optimizer_offload": False,
        "grad_offload": False,
        "forward_only": False,
        "dtype": "bfloat16",
        "use_dynamic_bsz": cfg.trainer.use_dynamic_bsz,
        "max_token_len_per_gpu": cfg.trainer.max_token_len_per_gpu,
        "use_remove_padding": cfg.trainer.use_remove_padding,
        "use_fused_kernels": False,
        "seed": 42,
        # Mcore-specific parallelism
        "tensor_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": None,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "sequence_parallel": True,
        "use_distributed_optimizer": True,
        "use_dist_checkpointing": False,
        "dist_checkpointing_path": None,
        "dist_ckpt_optim_fully_reshardable": False,
        "distrib_optim_fully_reshardable_mem_efficient": False,
        "use_mbridge": True,
        "vanilla_mbridge": True,
        "use_megatron_fsdp": False,
        "override_ddp_config": {},
        "override_transformer_config": {
            "recompute_granularity": "full",
            "recompute_modules": ["core_attn"],
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "override_mcore_model_config": {},
        "router_replay": {"mode": "disabled"},
    }
    return megatron


def _build_optimizer_config(
    trinity_optim: TrinityOptimizerConfig, strategy: str, total_training_steps: int
) -> dict:
    """Build veRL OptimizerConfig-compatible dict from Trinity's OptimizerConfig."""
    is_fsdp = strategy.startswith("fsdp")

    optim = {
        "lr": trinity_optim.lr,
        "lr_warmup_steps_ratio": trinity_optim.lr_warmup_steps_ratio,
        "lr_warmup_steps": trinity_optim.lr_warmup_steps,
        "total_training_steps": total_training_steps,
        "weight_decay": trinity_optim.weight_decay,
        "betas": list(trinity_optim.betas),
        "clip_grad": trinity_optim.clip_grad,
    }

    if is_fsdp:
        # FSDP uses FSDPOptimizerConfig
        optim["optimizer"] = _map_optimizer_name_fsdp(trinity_optim.optimizer_type)
        optim["optimizer_impl"] = "torch.optim"
        optim["min_lr_ratio"] = trinity_optim.min_lr_ratio
        optim["lr_scheduler_type"] = trinity_optim.lr_scheduler_type
        optim["num_cycles"] = 0.5
        optim["override_optimizer_config"] = None
        optim["zero_indexed_step"] = True
    else:
        # Megatron uses McoreOptimizerConfig
        optim["optimizer"] = trinity_optim.optimizer_type
        optim["lr_warmup_init"] = trinity_optim.min_lr_ratio * trinity_optim.lr
        optim["lr_decay_steps"] = total_training_steps
        optim["lr_decay_style"] = trinity_optim.lr_scheduler_type
        optim["min_lr"] = trinity_optim.min_lr_ratio * trinity_optim.lr
        optim["weight_decay_incr_style"] = "constant"
        optim["lr_wsd_decay_style"] = "exponential"
        optim["lr_wsd_decay_steps"] = None
        optim["use_checkpoint_opt_param_scheduler"] = False
        optim["override_optimizer_config"] = None

    return optim


def _build_critic_optimizer_config(strategy: str, total_training_steps: int) -> dict:
    """Build a default optimizer config for the critic model."""
    is_fsdp = strategy.startswith("fsdp")
    optim = {
        "lr": 1e-5,
        "lr_warmup_steps_ratio": 0.0,
        "lr_warmup_steps": -1,
        "total_training_steps": total_training_steps,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "clip_grad": 1.0,
    }
    if is_fsdp:
        optim["optimizer"] = "AdamW"
        optim["optimizer_impl"] = "torch.optim"
        optim["min_lr_ratio"] = 0.01
        optim["lr_scheduler_type"] = "constant"
        optim["num_cycles"] = 0.5
        optim["override_optimizer_config"] = None
        optim["zero_indexed_step"] = True
    else:
        optim["optimizer"] = "adam"
        optim["lr_warmup_init"] = 0.0
        optim["lr_decay_steps"] = total_training_steps
        optim["lr_decay_style"] = "constant"
        optim["min_lr"] = 0.0
        optim["weight_decay_incr_style"] = "constant"
        optim["lr_wsd_decay_style"] = "exponential"
        optim["lr_wsd_decay_steps"] = None
        optim["use_checkpoint_opt_param_scheduler"] = False
        optim["override_optimizer_config"] = None

    return optim


def _build_checkpoint_config(
    save_contents: Optional[List[str]] = None,
    load_contents: Optional[List[str]] = None,
) -> dict:
    """Build CheckpointConfig-compatible dict."""
    if save_contents is None:
        save_contents = ["model", "optimizer", "extra"]
    if load_contents is None:
        load_contents = ["model", "optimizer", "extra"]
    return {
        "save_contents": save_contents,
        "load_contents": load_contents,
        "async_save": False,
        "mbridge_config": {"distributed_filesystem": True, "memory_efficient": True},
        "strict": True,
    }


def _build_trainer_config(cfg: Config, total_training_steps: int) -> dict:
    """Build the trainer-level config dict (read by VERLTrainer)."""
    trainer = {
        "gpu_per_node": cfg.cluster.trainer_gpu_num_per_node,
        "node_num": cfg.cluster.trainer_node_num,
        "total_training_steps": total_training_steps,
        "sync_freq": cfg.synchronizer.sync_interval,
        "save_freq": cfg.trainer.save_interval,
        "default_local_dir": cfg.checkpoint_job_dir,
        "default_hdfs_dir": None,
        "del_local_ckpt_after_load": False,
    }
    if cfg.trainer.max_checkpoints_to_keep:
        trainer["max_actor_ckpt_to_keep"] = cfg.trainer.max_checkpoints_to_keep
        trainer["max_critic_ckpt_to_keep"] = cfg.trainer.max_checkpoints_to_keep
    else:
        trainer["max_actor_ckpt_to_keep"] = None
        trainer["max_critic_ckpt_to_keep"] = None

    if not cfg.continue_from_checkpoint:
        trainer["resume_mode"] = "disable"
    else:
        trainer["resume_mode"] = "auto"

    return trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_optimizer_name_fsdp(name: str) -> str:
    """Map Trinity's optimizer name to the class name FSDPOptimizerConfig expects."""
    mapping = {
        "adam": "AdamW",
        "adamw": "AdamW",
        "sgd": "SGD",
    }
    return mapping.get(name, name)
