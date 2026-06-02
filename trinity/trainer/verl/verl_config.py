"""veRL config builder for Trinity-RFT.

Converts Trinity's Config into a plain dict matching the structure that
veRL's ActorRolloutRefWorker.init_model() expects. The dict is consumed
via OmegaConf.create() → non-struct DictConfig → hydra instantiate.

No dataclass schemas, no OmegaConf.structured(), no _target_ inheritance issues.
"""
import math
import sys
from copy import deepcopy
from typing import Any

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.config import Config
from trinity.common.constants import EXPLORER_NAME
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def build_verl_config(config: Config) -> dict:
    """Convert Trinity Config into veRL-compatible config dict.

    The output dict will be wrapped with OmegaConf.create() to produce a
    non-struct DictConfig. veRL's init_model() then calls
    omega_conf_to_dataclass() which uses hydra instantiate on nodes with
    _target_.

    Only _target_ on top-level nodes (model, actor, ref) — nested configs
    (engine, optim, checkpoint) are passed as plain kwargs.
    """
    algo_config = config.algorithm
    trainer_config = config.trainer
    cluster_config = config.cluster
    model_config = config.model
    buffer_config = config.buffer

    algorithm = ALGORITHM_TYPE.get(algo_config.algorithm_type)
    rollout_n = algo_config.repeat_times
    strategy = trainer_config.trainer_strategy
    use_dynamic_bsz = trainer_config.use_dynamic_bsz
    use_remove_padding = trainer_config.use_remove_padding
    world_size = cluster_config.trainer_gpu_num
    sp_size = trainer_config.ulysses_sequence_parallel_size or 1
    train_batch_size = buffer_config.train_batch_size

    # Auto-compute max_token_len_per_gpu if not set
    max_token_len = trainer_config.max_token_len_per_gpu
    if max_token_len is None:
        max_token_len = math.ceil(model_config.max_model_len / sp_size)

    # Optimizer (shared between actor and critic with different defaults)
    optim = algo_config.optimizer
    optim_map = {"adam": "AdamW", "adamw": "AdamW", "sgd": "SGD"}
    optimizer_name = optim_map.get(optim.optimizer_type, optim.optimizer_type) if strategy.startswith("fsdp") else optim.optimizer_type
    total_training_steps = trainer_config.total_steps or sys.maxsize

    # Temperature
    temperature = (
        buffer_config.explorer_input.tasksets[0].rollout_args.temperature
        if buffer_config.explorer_input.tasksets
        else 1.0
    )

    # --- Model ---
    override_config: dict[str, Any] = {}
    if model_config.rope_scaling is not None:
        override_config["rope_scaling"] = model_config.rope_scaling
    if model_config.rope_theta is not None:
        override_config["rope_theta"] = model_config.rope_theta

    model = {
        "_target_": "verl.workers.config.HFModelConfig",
        "path": model_config.model_path,
        "trust_remote_code": model_config.trust_remote_code,
        "custom_chat_template": model_config.custom_chat_template,
        "use_remove_padding": use_remove_padding,
        "use_fused_kernels": False,
        "fused_kernel_options": {},
        "enable_gradient_checkpointing": True,
        "enable_activation_offload": False,
        "use_shm": False,
        "override_config": override_config,
        "mtp": {"_target_": "verl.workers.config.MtpConfig", "enable": False},
    }

    # LoRA
    if model_config.lora_configs is not None:
        lora = model_config.lora_configs[0]
        model["lora_rank"] = lora.lora_rank
        model["lora_alpha"] = lora.lora_alpha
        model["target_modules"] = lora.target_modules
        model["exclude_modules"] = lora.exclude_modules
        if not lora.is_dummy:
            model["lora_adapter_path"] = lora.path

    # --- Engine (FSDP) ---
    fsdp_size = world_size  # default: all GPUs in one FSDP group
    engine = {
        "strategy": strategy,
        "fsdp_size": fsdp_size,
        "ulysses_sequence_parallel_size": sp_size,
        "param_offload": False,
        "optimizer_offload": False,
        "offload_policy": False,
        "reshard_after_forward": True,
        "forward_prefetch": False,
        "dtype": "bfloat16",
        "model_dtype": "fp32",
    }

    # --- Actor ---
    actor = {
        "_target_": "verl.workers.config.ActorConfig",
        "strategy": strategy,
        "rollout_n": rollout_n,
        "ppo_mini_batch_size": train_batch_size,
        "ppo_micro_batch_size_per_gpu": 1,
        "use_dynamic_bsz": use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": max_token_len,
        "ppo_epochs": 1,
        "shuffle": False,
        "loss_agg_mode": algo_config.loss_agg_mode or "token-mean",
        "entropy_coeff": 0,
        "use_kl_loss": algo_config.kl_loss_fn not in (None, "none"),
        "use_prefix_grouper": False,
        "freeze_vision_tower": False,
        "engine": deepcopy(engine),
        "optim": {
            "lr": optim.lr,
            "lr_warmup_steps": optim.lr_warmup_steps,
            "lr_warmup_steps_ratio": optim.lr_warmup_steps_ratio,
            "lr_scheduler_type": optim.lr_scheduler_type,
            "total_training_steps": total_training_steps,
            "betas": list(optim.betas),
            "clip_grad": optim.clip_grad,
            "weight_decay": optim.weight_decay,
            "min_lr_ratio": optim.min_lr_ratio,
            "optimizer": optimizer_name,
        },
        "checkpoint": {
            "save_contents": ["model", "optimizer", "extra"],
            "load_contents": ["model", "optimizer", "extra"],
            "mbridge_config": {"distributed_filesystem": True, "memory_efficient": True},
        },
    }

    # --- Ref ---
    ref_engine = deepcopy(engine)
    ref_engine["forward_only"] = True
    ref = {
        "_target_": "verl.workers.config.ActorConfig",
        "strategy": strategy,
        "rollout_n": rollout_n,
        "ppo_mini_batch_size": train_batch_size,
        "use_dynamic_bsz": use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": max_token_len,
        "engine": ref_engine,
        "optim": None,
        "checkpoint": {
            "save_contents": ["model"],
            "load_contents": ["model"],
            "mbridge_config": {"distributed_filesystem": True, "memory_efficient": True},
        },
        # These are popped/remapped by veRL's init_model before instantiation
        "log_prob_micro_batch_size_per_gpu": 1,
        "log_prob_use_dynamic_bsz": use_dynamic_bsz,
        "log_prob_max_token_len_per_gpu": max_token_len,
    }

    # --- Rollout (log_prob settings for hybrid engine) ---
    rollout = {
        "temperature": temperature,
        "n": rollout_n,
        "log_prob_use_dynamic_bsz": use_dynamic_bsz,
        "log_prob_micro_batch_size_per_gpu": 1,
        "log_prob_max_token_len_per_gpu": max_token_len,
    }

    # --- Critic ---
    critic_optimizer_name = optim_map.get("adam", "AdamW") if strategy.startswith("fsdp") else "adam"
    critic = {
        "enable": algorithm.use_critic,
        "strategy": strategy,
        "ppo_mini_batch_size": train_batch_size,
        "ppo_micro_batch_size_per_gpu": 1,
        "ppo_infer_micro_batch_size_per_gpu": 1,
        "ppo_infer_max_token_len_per_gpu": max_token_len,
        "use_dynamic_bsz": use_dynamic_bsz,
        "ppo_max_token_len_per_gpu": max_token_len,
        "forward_max_token_len_per_gpu": max_token_len,
        "ppo_epochs": 1,
        "rollout_n": rollout_n,
        "loss_agg_mode": "token-mean",
        "cliprange_value": 0.0,
        "model": {
            "path": model_config.critic_model_path,
            "tokenizer_path": model_config.critic_model_path,
            "trust_remote_code": model_config.trust_remote_code,
            "enable_gradient_checkpointing": True,
            "override_config": deepcopy(override_config),
        },
        "optim": {
            "lr": 1e-6,
            "clip_grad": optim.clip_grad,
            "total_training_steps": total_training_steps,
            "optimizer": critic_optimizer_name,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
        },
        "checkpoint": {
            "save_contents": ["model", "optimizer", "extra"],
            "load_contents": ["model", "optimizer", "extra"],
            "mbridge_config": {"distributed_filesystem": True, "memory_efficient": True},
        },
        # engine config for critic (used by verl_trainer.py to build TrainingWorkerConfig)
        "engine": deepcopy(engine),
    }

    # --- Trainer ---
    trainer = {
        "nnodes": cluster_config.trainer_node_num,
        "n_gpus_per_node": cluster_config.trainer_gpu_num_per_node,
        "total_training_steps": total_training_steps,
        "save_freq": trainer_config.save_interval,
        "default_local_dir": config.checkpoint_job_dir,
        "default_hdfs_dir": None,
        "resume_mode": "auto" if config.continue_from_checkpoint else "disable",
        "resume_from_path": "",
        "del_local_ckpt_after_load": False,
        "max_actor_ckpt_to_keep": trainer_config.max_checkpoints_to_keep,
        "max_critic_ckpt_to_keep": trainer_config.max_checkpoints_to_keep,
        "balance_batch": True,
        "critic_warmup": 0,
        "device": "cuda",
    }

    # --- Algorithm (for verl_trainer driver-side logic) ---
    algorithm_dict = {
        "use_kl_in_reward": algo_config.kl_penalty_fn not in (None, "none"),
    }

    # --- Data ---
    data = {
        "train_batch_size": train_batch_size,
        "trust_remote_code": model_config.trust_remote_code,
    }

    # --- Assemble ---
    result = {
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": model,
            "actor": actor,
            "ref": ref,
            "rollout": rollout,
            "nccl_timeout": config.synchronizer.sync_timeout,
        },
        "critic": critic,
        "trainer": trainer,
        "algorithm": algorithm_dict,
        "data": data,
        "global_profiler": {},
    }

    # --- Apply user overlay from trainer_config ---
    user_overlay = trainer_config.trainer_config
    if user_overlay and isinstance(user_overlay, dict):
        result = _deep_merge(result, user_overlay)

    return result
