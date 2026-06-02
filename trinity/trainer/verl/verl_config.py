"""veRL config dataclasses for Trinity-RFT.

These dataclasses serve as OmegaConf schemas with Trinity's defaults.
They are intentionally minimal — only fields that Trinity's synchronize_config()
actually writes are declared here. When omega_conf_to_dataclass() is called
downstream (via _target_), veRL's full dataclass provides defaults for any
field not listed here.

Flow:
    Trinity dataclass (schema + defaults)
        → OmegaConf.structured()
        → DictConfig (mutable, incrementally populated by synchronize_config)
        → omega_conf_to_dataclass() + _target_
        → veRL dataclass (used by engine workers)
"""
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from verl.workers.config import (
    MtpConfig as _MtpConfigBase,
    PolicyLossConfig as _PolicyLossConfigBase,
    RouterReplayConfig as _RouterReplayConfigBase,
)

from trinity.algorithm import ALGORITHM_TYPE
from trinity.common.config import Config, SynchronizerConfig, set_if_none
from trinity.common.constants import EXPLORER_NAME
from trinity.common.patch import kimi_vl_monkey_patch_decorator
from trinity.utils.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Nested configs
# ---------------------------------------------------------------------------


@dataclass
class Data:
    train_batch_size: int = 1024
    trust_remote_code: bool = False


@dataclass
class FSDPEngineConfig:
    """OmegaConf-compatible schema for verl.workers.config.FSDPEngineConfig.

    Declared from scratch (not inheriting) because veRL's BaseConfig uses
    loose typing (e.g. `int = None`) that OmegaConf.structured() rejects.
    Only fields Trinity actually reads/writes are listed here; hydra
    instantiate will fill the rest from veRL's defaults via _target_.
    """

    _target_: str = "verl.workers.config.FSDPEngineConfig"
    strategy: str = "fsdp"
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    forward_only: bool = False
    reshard_after_forward: bool = True
    wrap_policy: Dict[str, Any] = field(default_factory=dict)
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    dtype: str = "bfloat16"
    mixed_precision: Optional[Dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False




@dataclass
class Optim:
    optimizer: str = "adam"
    optimizer_impl: str = "torch.optim"
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = 0.0
    lr_scheduler_type: str = "constant"
    total_training_steps: int = -1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    clip_grad: float = 1.0
    lr_warmup_init: Optional[float] = None
    lr_decay_steps: Optional[int] = None
    lr_decay_style: Optional[str] = None
    min_lr: Optional[float] = None
    weight_decay: float = 0.01
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class CheckpointConfig:
    """OmegaConf-compatible schema for verl.trainer.config.CheckpointConfig."""

    _target_: str = "verl.trainer.config.CheckpointConfig"
    save_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    mbridge_config: Dict[str, Any] = field(
        default_factory=lambda: dict(distributed_filesystem=True, memory_efficient=True)
    )


@dataclass
class RefCheckpointConfig:
    _target_: str = "verl.trainer.config.CheckpointConfig"
    save_contents: List[str] = field(default_factory=lambda: ["model"])
    load_contents: List[str] = field(default_factory=lambda: ["model"])
    async_save: bool = False
    mbridge_config: Dict[str, Any] = field(
        default_factory=lambda: dict(distributed_filesystem=True, memory_efficient=True)
    )


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class MtpConfig(_MtpConfigBase):
    """MtpConfig with _target_ for hydra instantiation (veRL's BaseConfig defaults to "")."""

    _target_: str = "verl.workers.config.MtpConfig"


@dataclass
class PolicyLossConfig(_PolicyLossConfigBase):
    _target_: str = "verl.workers.config.PolicyLossConfig"


@dataclass
class RouterReplayConfig(_RouterReplayConfigBase):
    _target_: str = "verl.workers.config.RouterReplayConfig"


@dataclass
class ActorModel:
    """Maps to verl.workers.config.model.HFModelConfig via _target_."""

    _target_: str = "verl.workers.config.model.HFModelConfig"
    path: str = ""
    external_lib: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    use_remove_padding: Optional[bool] = None
    use_fused_kernels: bool = False
    fused_kernel_options: Dict[str, Any] = field(default_factory=dict)
    custom_chat_template: Optional[str] = None
    enable_activation_offload: bool = False
    use_shm: bool = False
    trust_remote_code: bool = False
    lora_rank: int = 0
    lora_alpha: int = 32
    target_modules: Optional[str] = "all-linear"
    exclude_modules: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    mtp: MtpConfig = field(default_factory=MtpConfig)


# ---------------------------------------------------------------------------
# Actor / Ref / Rollout / Critic
# ---------------------------------------------------------------------------


@dataclass
class Actor:
    """Maps to verl.workers.config.actor.ActorConfig via _target_."""

    _target_: str = "verl.workers.config.actor.ActorConfig"
    strategy: Optional[str] = None
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = 1
    use_dynamic_bsz: Optional[bool] = None
    ppo_max_token_len_per_gpu: Optional[int] = None
    ppo_epochs: int = 1
    shuffle: bool = False
    freeze_vision_tower: bool = False
    use_prefix_grouper: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: Optim = field(default_factory=Optim)
    engine: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    profiler: dict = field(default_factory=dict)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    rollout_n: int = 1
    data_loader_seed: Optional[int] = None
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[float] = None
    entropy_coeff: float = 0
    use_kl_loss: bool = False


@dataclass
class Ref:
    """Maps to verl.workers.config.actor.ActorConfig via _target_ (forward_only engine)."""

    _target_: str = "verl.workers.config.actor.ActorConfig"
    strategy: Optional[str] = None
    engine: FSDPEngineConfig = field(default_factory=lambda: FSDPEngineConfig(forward_only=True))
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = 1
    log_prob_use_dynamic_bsz: Optional[bool] = None
    log_prob_max_token_len_per_gpu: Optional[int] = None
    checkpoint: RefCheckpointConfig = field(default_factory=RefCheckpointConfig)
    optim: Optional[Optim] = None
    profiler: dict = field(default_factory=dict)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    rollout_n: int = 1


@dataclass
class Rollout:
    temperature: float = 1.0
    n: int = 1
    log_prob_use_dynamic_bsz: Optional[bool] = None
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_max_token_len_per_gpu: Optional[int] = None


@dataclass
class ActorRolloutRef:
    hybrid_engine: bool = True
    model: ActorModel = field(default_factory=ActorModel)
    actor: Actor = field(default_factory=Actor)
    ref: Ref = field(default_factory=Ref)
    rollout: Rollout = field(default_factory=Rollout)
    nccl_timeout: float = 600
    synchronizer: Optional[SynchronizerConfig] = None
    explorer_name: str = EXPLORER_NAME


@dataclass
class CriticModel:
    path: str = ""
    tokenizer_path: str = ""
    override_config: Dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
    enable_gradient_checkpointing: bool = True
    use_remove_padding: Optional[bool] = None
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    freeze_vision_tower: bool = False


@dataclass
class Critic:
    enable: bool = False
    strategy: Optional[str] = None
    optim: Optim = field(default_factory=Optim)
    model: CriticModel = field(default_factory=CriticModel)
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_max_token_len_per_gpu: Optional[int] = None
    use_dynamic_bsz: Optional[bool] = None
    ppo_max_token_len_per_gpu: Optional[int] = None
    forward_max_token_len_per_gpu: Optional[int] = None
    ppo_epochs: int = 1
    shuffle: bool = False
    cliprange_value: float = 0.0
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    rollout_n: int = 1
    loss_agg_mode: str = "token-mean"
    data_loader_seed: int = 42
    nccl_timeout: float = 600
    ray_namespace: str = ""
    profiler: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm / Reward / Trainer
# ---------------------------------------------------------------------------


@dataclass
class KL_Ctrl:
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: float = 10000
    target_kl: float = 0.1


@dataclass
class Algorithm:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KL_Ctrl = field(default_factory=KL_Ctrl)


@dataclass
class Trainer:
    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: Optional[int] = None
    project_name: str = ""
    group_name: str = ""
    experiment_name: str = ""
    logger: List[str] = field(default_factory=list)
    nnodes: int = 0
    n_gpus_per_node: int = 0
    save_freq: int = 0
    resume_mode: str = "auto"
    resume_from_path: str = ""
    test_freq: int = 0
    critic_warmup: int = 0
    default_hdfs_dir: Optional[str] = None
    del_local_ckpt_after_load: bool = False
    default_local_dir: str = ""
    sync_freq: int = 0
    max_actor_ckpt_to_keep: Optional[int] = None
    max_critic_ckpt_to_keep: Optional[int] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class veRLConfig:
    data: Data = field(default_factory=Data)
    actor_rollout_ref: ActorRolloutRef = field(default_factory=ActorRolloutRef)
    critic: Critic = field(default_factory=Critic)
    algorithm: Algorithm = field(default_factory=Algorithm)
    trainer: Trainer = field(default_factory=Trainer)
    global_profiler: dict = field(default_factory=dict)
    synchronizer: Optional[SynchronizerConfig] = None
    enable_preview: bool = True

    @kimi_vl_monkey_patch_decorator
    def _check_parallel_config(
        self,
        component_name: str,
        model_config: Union[ActorModel, CriticModel],
        engine_config: FSDPEngineConfig,
        train_batch_size: int,
        world_size: int,
    ) -> None:
        strategy = engine_config.strategy
        if not strategy or not strategy.startswith("fsdp"):
            return

        sp_size = engine_config.ulysses_sequence_parallel_size
        if sp_size < 1:
            sp_size = 1
            engine_config.ulysses_sequence_parallel_size = sp_size

        if world_size % sp_size != 0:
            raise ValueError(
                f"The number of trainer GPUs ({world_size}) must be "
                f"divisible by `{component_name}.engine.ulysses_sequence_parallel_size` ({sp_size})."
            )
        if train_batch_size % (world_size // sp_size) != 0:
            raise ValueError(
                f"The batch size ({train_batch_size}) must be divisible by "
                f"the number of GPUs ({world_size}) divided by the sequence "
                f"parallelism size ({sp_size})."
            )

        try:
            import transformers

            hf_config = transformers.AutoConfig.from_pretrained(
                model_config.path, trust_remote_code=model_config.trust_remote_code
            )
            num_attention_heads = hf_config.num_attention_heads
        except Exception:
            num_attention_heads = None

        if num_attention_heads and num_attention_heads % sp_size != 0:
            raise ValueError(
                f"The number of attention heads ({num_attention_heads}) must be "
                f"divisible by ulysses_sequence_parallel_size ({sp_size})."
            )

        fsdp_size = engine_config.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            engine_config.fsdp_size = world_size
        elif world_size % fsdp_size != 0:
            raise ValueError(
                f"The number of GPUs ({world_size}) must be "
                f"divisible by `{component_name}.engine.fsdp_size` ({fsdp_size})."
            )

    def _adjust_token_len_if_needed(
        self,
        obj,
        engine_config: FSDPEngineConfig,
        config: Config,
        component_name: str,
        token_len_attr: str = "ppo_max_token_len_per_gpu",
    ) -> None:
        current_token_len = getattr(obj, token_len_attr)
        if current_token_len is None:
            return
        sp = engine_config.ulysses_sequence_parallel_size
        required_min = config.model.max_model_len
        if required_min is None:
            raise ValueError(
                "config.model.max_model_len must be set to adjust token length based on SP size."
            )

        if current_token_len * sp < required_min:
            new_token_len = math.ceil(required_min / sp)
            setattr(obj, token_len_attr, new_token_len)
            logger.warning(
                f"{component_name}.{token_len_attr} adjusted to {new_token_len} "
                f"to match model.max_model_len ({required_min})."
            )

    def synchronize_config(self, config: Config) -> None:  # noqa: C901
        """Populate veRL config from Trinity's global config."""
        # --- Trainer ---
        self.trainer.nnodes = config.cluster.trainer_node_num
        self.trainer.n_gpus_per_node = config.cluster.trainer_gpu_num_per_node
        self.trainer.total_training_steps = config.trainer.total_steps or sys.maxsize
        self.trainer.sync_freq = config.synchronizer.sync_interval
        self.trainer.save_freq = config.trainer.save_interval
        self.trainer.project_name = config.project
        self.trainer.group_name = config.group
        self.trainer.experiment_name = config.name
        self.trainer.default_local_dir = config.checkpoint_job_dir
        if config.trainer.max_checkpoints_to_keep is not None:
            self.trainer.max_actor_ckpt_to_keep = config.trainer.max_checkpoints_to_keep
            self.trainer.max_critic_ckpt_to_keep = config.trainer.max_checkpoints_to_keep
        self.trainer.resume_mode = "auto" if config.continue_from_checkpoint else "disable"

        # --- Global ---
        self.data.train_batch_size = config.buffer.train_batch_size
        self.data.trust_remote_code = config.model.trust_remote_code
        self.synchronizer = config.synchronizer
        self.actor_rollout_ref.nccl_timeout = config.synchronizer.sync_timeout
        self.actor_rollout_ref.synchronizer = config.synchronizer
        self.actor_rollout_ref.explorer_name = config.explorer.name

        algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
        self.critic.enable = algorithm.use_critic
        self.critic.nccl_timeout = config.synchronizer.sync_timeout
        self.critic.ray_namespace = config.synchronizer.ray_namespace

        # --- Actor Model ---
        actor_model = self.actor_rollout_ref.model
        actor_model.path = config.model.model_path
        actor_model.trust_remote_code = config.model.trust_remote_code
        actor_model.custom_chat_template = config.model.custom_chat_template
        if config.model.rope_scaling is not None:
            actor_model.override_config["rope_scaling"] = config.model.rope_scaling
        if config.model.rope_theta is not None:
            actor_model.override_config["rope_theta"] = config.model.rope_theta

        # --- Actor ---
        actor = self.actor_rollout_ref.actor
        actor.ppo_mini_batch_size = config.buffer.train_batch_size
        actor.rollout_n = config.algorithm.repeat_times
        actor.optim.total_training_steps = self.trainer.total_training_steps
        if config.trainer.grad_clip is not None:
            actor.optim.clip_grad = config.trainer.grad_clip
        for attr, trainer_attr in [
            ("ppo_max_token_len_per_gpu", "max_token_len_per_gpu"),
            ("strategy", "trainer_strategy"),
        ]:
            set_if_none(actor, attr, getattr(config.trainer, trainer_attr))

        # Engine config
        sp_size = config.trainer.ulysses_sequence_parallel_size
        if sp_size is not None:
            actor.engine.ulysses_sequence_parallel_size = sp_size
        if actor.strategy is not None:
            actor.engine.strategy = actor.strategy

        self._check_parallel_config(
            component_name="actor",
            model_config=actor_model,
            engine_config=actor.engine,
            train_batch_size=config.buffer.train_batch_size,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=actor, engine_config=actor.engine, config=config, component_name="actor"
        )

        # --- Rollout ---
        rollout = self.actor_rollout_ref.rollout
        rollout.temperature = (
            config.buffer.explorer_input.tasksets[0].rollout_args.temperature
            if config.buffer.explorer_input.tasksets
            else 1.0
        )
        rollout.n = config.algorithm.repeat_times

        # --- Ref ---
        ref = self.actor_rollout_ref.ref
        for ref_attr, trainer_attr in [
            ("log_prob_use_dynamic_bsz", "use_dynamic_bsz"),
            ("log_prob_max_token_len_per_gpu", "max_token_len_per_gpu"),
            ("strategy", "trainer_strategy"),
        ]:
            set_if_none(ref, ref_attr, getattr(config.trainer, trainer_attr))
        if sp_size is not None:
            ref.engine.ulysses_sequence_parallel_size = sp_size
        if ref.strategy is not None:
            ref.engine.strategy = ref.strategy
        ref.engine.forward_only = True
        ref.rollout_n = config.algorithm.repeat_times

        self._check_parallel_config(
            component_name="ref",
            model_config=actor_model,
            engine_config=ref.engine,
            train_batch_size=config.buffer.train_batch_size,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=ref,
            engine_config=ref.engine,
            config=config,
            component_name="ref",
            token_len_attr="log_prob_max_token_len_per_gpu",
        )

        # --- Critic ---
        critic = self.critic
        critic.model.path = config.model.critic_model_path
        critic.model.tokenizer_path = config.model.critic_model_path
        if config.model.rope_scaling is not None:
            critic.model.override_config["rope_scaling"] = config.model.rope_scaling
        if config.model.rope_theta is not None:
            critic.model.override_config["rope_theta"] = config.model.rope_theta
        critic.ppo_mini_batch_size = config.buffer.train_batch_size
        critic.rollout_n = config.algorithm.repeat_times
        critic.optim.total_training_steps = self.trainer.total_training_steps
        if config.trainer.grad_clip is not None:
            critic.optim.clip_grad = config.trainer.grad_clip
        for attr, trainer_attr in [
            ("strategy", "trainer_strategy"),
            ("ppo_max_token_len_per_gpu", "max_token_len_per_gpu"),
        ]:
            set_if_none(critic, attr, getattr(config.trainer, trainer_attr))
        if sp_size is not None:
            critic.model.fsdp_config.ulysses_sequence_parallel_size = sp_size
        if critic.strategy is not None:
            critic.model.fsdp_config.strategy = critic.strategy

        self._check_parallel_config(
            component_name="critic",
            model_config=critic.model,
            engine_config=critic.model.fsdp_config,
            train_batch_size=config.buffer.train_batch_size,
            world_size=config.cluster.trainer_gpu_num,
        )
        self._adjust_token_len_if_needed(
            obj=critic,
            engine_config=critic.model.fsdp_config,
            config=config,
            component_name="critic",
        )
        set_if_none(critic, "forward_max_token_len_per_gpu", critic.ppo_max_token_len_per_gpu)

        # --- LoRA ---
        if config.model.lora_configs is not None:
            lora_config = config.model.lora_configs[0]
            for attr in ["lora_rank", "lora_alpha", "target_modules", "exclude_modules"]:
                setattr(actor_model, attr, getattr(lora_config, attr))
            if not lora_config.is_dummy:
                actor_model.lora_adapter_path = lora_config.path
            if actor.strategy not in ["fsdp", "fsdp2"]:
                logger.warning(f"Lora requires fsdp/fsdp2, got {actor.strategy}, changing to fsdp.")
                actor.strategy = "fsdp"
                actor.engine.strategy = "fsdp"

        # --- Algorithm / optimizer ---
        optim_config = config.algorithm.optimizer
        actor_optim = actor.optim
        for field_name in optim_config.__dataclass_fields__:
            field_value = getattr(optim_config, field_name)
            if field_name == "optimizer_type":
                actor_optim.optimizer = field_value
            elif hasattr(actor_optim, field_name):
                setattr(actor_optim, field_name, field_value)

        set_if_none(actor_optim, "lr_warmup_init", optim_config.min_lr_ratio * optim_config.lr)
        set_if_none(actor_optim, "lr_decay_steps", self.trainer.total_training_steps)
        set_if_none(actor_optim, "lr_decay_style", optim_config.lr_scheduler_type)
        set_if_none(actor_optim, "min_lr_ratio", optim_config.min_lr_ratio)
        set_if_none(actor_optim, "min_lr", optim_config.min_lr_ratio * optim_config.lr)

        critic_optim = critic.optim
        set_if_none(critic_optim, "lr_warmup_init", critic_optim.lr * 0.01)
        set_if_none(critic_optim, "lr_decay_steps", self.trainer.total_training_steps)
        set_if_none(critic_optim, "lr_decay_style", "constant")
        set_if_none(critic_optim, "min_lr_ratio", 0.01)
        set_if_none(critic_optim, "min_lr", critic_optim.lr * 0.01)

        if config.trainer.trainer_strategy.startswith("fsdp"):
            optim_map = {"adam": "AdamW", "adamw": "AdamW", "sgd": "SGD"}
            actor_optim.optimizer = optim_map.get(actor_optim.optimizer, actor_optim.optimizer)
            critic_optim.optimizer = optim_map.get(critic_optim.optimizer, critic_optim.optimizer)

        actor.use_kl_loss = config.algorithm.kl_loss_fn != "none"
        self.algorithm.use_kl_in_reward = config.algorithm.kl_penalty_fn != "none"

        if config.algorithm.algorithm_type == "dpo":
            logger.warning("DPO micro batch size is doubled for computing loss.")
            actor.ppo_micro_batch_size_per_gpu = (actor.ppo_micro_batch_size_per_gpu or 1) * 2
            ref.log_prob_micro_batch_size_per_gpu = (ref.log_prob_micro_batch_size_per_gpu or 1) * 2

        # Rollout log_prob config (for lora: reuses actor settings)
        for rollout_attr, actor_attr in [
            ("log_prob_use_dynamic_bsz", "use_dynamic_bsz"),
            ("log_prob_micro_batch_size", "ppo_micro_batch_size"),
            ("log_prob_micro_batch_size_per_gpu", "ppo_micro_batch_size_per_gpu"),
            ("log_prob_max_token_len_per_gpu", "ppo_max_token_len_per_gpu"),
        ]:
            set_if_none(rollout, rollout_attr, getattr(actor, actor_attr))

        # --- use_dynamic_bsz / use_remove_padding propagation ---
        use_dynamic_bsz = config.trainer.use_dynamic_bsz
        actor.use_dynamic_bsz = use_dynamic_bsz
        critic.use_dynamic_bsz = use_dynamic_bsz

        use_remove_padding = config.trainer.use_remove_padding
        actor_model.use_remove_padding = use_remove_padding

        self.enable_preview = config.trainer.enable_preview
