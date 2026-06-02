# -*- coding: utf-8 -*-
"""veRL Trainer Class (engine-based)

Standalone trainer wrapper for Trinity-RFT, using veRL's engine-based worker
architecture. Does NOT inherit from veRL's RayPPOTrainer (deprecated) or
PPOTrainer (includes rollout management that conflicts with Trinity's
decoupled Explorer/Buffer/Trainer architecture).

Only the training-step mechanics are used: compute_log_prob, update_actor,
update_critic, update_weights.
"""
import asyncio
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import ray
import torch
import transformers
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.workers.config import EngineConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE, KL_FN
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.constants import SaveStrategy
from trinity.common.experience import Experience
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.trainer.verl.utils import compute_data_metrics, to_data_proto
from trinity.trainer.verl.verl_config import build_verl_config
from trinity.trainer.verl.workers import TrinityActorRolloutRefWorker
from trinity.utils.log import get_logger


class CheckpointMonitor:
    def __init__(
        self, save_strategy: SaveStrategy, default_local_dir: str, default_hdfs_dir: str = None
    ):
        self.logger = get_logger("checkpoint_monitor", in_ray_actor=True)
        self.default_local_dir = default_local_dir
        self.default_hdfs_dir = default_hdfs_dir
        self.local_latest_checkpointed_iteration = os.path.join(
            default_local_dir, "latest_checkpointed_iteration.txt"
        )
        self.local_latest_state_dict_iteration = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        self.checkpoint_counter = defaultdict(int)
        self.state_dict_counter = defaultdict(int)
        self.checkpoint_steps = set()
        self.state_dict_steps = set()
        self.latest_checkpoint_step = 0
        self.latest_state_dict_step = 0

        self.save_strategy = save_strategy
        self.condition = asyncio.Condition()
        self.current_identifier = 0
        self.saving_count = 0

    def update_latest_checkpoint_step(self, step: int):
        assert step >= self.latest_checkpoint_step
        if step == self.latest_checkpoint_step:
            return
        self.latest_checkpoint_step = step
        with open(self.local_latest_checkpointed_iteration, "w") as f:
            f.write(str(step))
        if step in self.state_dict_counter:
            assert self.state_dict_counter[step] == 0
            self.update_latest_state_dict_step(step)

        if self.default_hdfs_dir is not None:
            local_path = os.path.join(self.default_local_dir, f"global_step_{step}")
            hdfs_path = os.path.join(self.default_hdfs_dir, f"global_step_{step}")
            self.logger.info(f"Uploading checkpoint to {hdfs_path}")
            from verl.utils import hdfs_io

            hdfs_io.makedirs(hdfs_path, exist_ok=True)
            hdfs_io.copy(src=local_path, dst=hdfs_path, dirs_exist_ok=True)
        self.logger.info(f"Checkpoint at step {step} saved.")

    def update_latest_state_dict_step(self, step: int):
        assert step >= self.latest_state_dict_step
        if step == self.latest_state_dict_step:
            return
        self.latest_state_dict_step = step
        with open(self.local_latest_state_dict_iteration, "w") as f:
            f.write(str(step))

    async def register_thread_count(
        self,
        step: int,
        *,
        state_dict_thread_count: int = 0,
        checkpoint_thread_count: int = 0,
    ):
        if state_dict_thread_count != 0:
            self.state_dict_counter[step] += state_dict_thread_count
        if checkpoint_thread_count != 0:
            self.checkpoint_counter[step] += checkpoint_thread_count

    async def monitor_step(self, step: int, is_state_dict: bool = False):
        if is_state_dict:
            self.state_dict_steps.add(step)
            if self.state_dict_counter[step] == 0:
                self.update_latest_state_dict_step(step)
        else:
            self.checkpoint_steps.add(step)
            if self.checkpoint_counter[step] == 0 and self.state_dict_counter[step] == 0:
                self.update_latest_checkpoint_step(step)

    async def notify_started(self, node_id: str, job_id: str):
        if self.save_strategy == SaveStrategy.SINGLE_THREAD:
            identifier = self.current_identifier + 1
        elif self.save_strategy == SaveStrategy.SINGLE_PROCESS:
            identifier = f"{node_id}_{job_id}"
        elif self.save_strategy == SaveStrategy.SINGLE_NODE:
            identifier = node_id
        elif self.save_strategy == SaveStrategy.UNRESTRICTED:
            return
        else:
            raise ValueError(f"Invalid save strategy: {self.save_strategy}")

        async with self.condition:
            if identifier != self.current_identifier and self.saving_count > 0:
                await self.condition.wait_for(lambda: self.saving_count == 0)
            self.current_identifier = identifier
            self.saving_count += 1

    async def notify_finished(self, step: int, is_state_dict: bool = False):
        async with self.condition:
            self.saving_count -= 1
            self.condition.notify_all()
        if is_state_dict:
            self.state_dict_counter[step] -= 1
            if (
                step in self.state_dict_steps or step in self.checkpoint_steps
            ) and self.state_dict_counter[step] == 0:
                self.update_latest_state_dict_step(step)
                if step in self.checkpoint_steps and self.checkpoint_counter[step] == 0:
                    self.update_latest_checkpoint_step(step)
        else:
            self.checkpoint_counter[step] -= 1
            if (
                step in self.checkpoint_steps
                and self.checkpoint_counter[step] == 0
                and self.state_dict_counter[step] == 0
            ):
                self.update_latest_checkpoint_step(step)

    @classmethod
    def get_actor(
        cls,
        namespace: str,
        save_strategy: Optional[SaveStrategy] = None,
        default_local_dir: Optional[str] = None,
        default_hdfs_dir: Optional[str] = None,
    ):
        return (
            ray.remote(cls)
            .options(
                name="checkpoint_monitor",
                namespace=namespace,
                get_if_exists=True,
            )
            .remote(
                save_strategy=save_strategy,
                default_local_dir=default_local_dir,
                default_hdfs_dir=default_hdfs_dir,
            )
        )


class VerlPPOTrainerWrapper(TrainEngineWrapper):
    """Standalone veRL trainer wrapper for Trinity-RFT.

    Manages engine-based worker groups for actor/critic/ref training without
    inheriting from veRL's deprecated RayPPOTrainer or its new PPOTrainer
    (which bundles rollout logic that Trinity handles separately).
    """

    def __init__(self, global_config: Config):
        self.logger = get_logger(__name__, in_ray_actor=True)
        self.logger.info(
            f"Initializing verl Trainer with {global_config.trainer.trainer_strategy} backend"
        )
        self.config = OmegaConf.create(build_verl_config(global_config))

        local_path = copy_local_path_from_hdfs(self.config.actor_rollout_ref.model.path)

        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        hf_config = transformers.AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        with init_empty_weights():
            self.empty_model = transformers.AutoModel.from_config(
                hf_config, trust_remote_code=trust_remote_code
            )

        # Determine which components are needed
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)

        # LoRA ref-in-actor detection
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = (
            lora_rank > 0
            or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # Worker class selection
        ActorRolloutRefWorkerCls = TrinityActorRolloutRefWorker
        CriticWorkerCls = TrainingWorker

        self.checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=global_config.synchronizer.ray_namespace,
            save_strategy=global_config.trainer.save_strategy,
            default_local_dir=self.config.trainer.default_local_dir,
            default_hdfs_dir=self.config.trainer.default_hdfs_dir,
        )

        # Resource pool setup
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        self.role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorkerCls),
            Role.Critic: ray.remote(CriticWorkerCls),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorkerCls),
        }

        # Algorithm setup
        self.algorithm_config = global_config.algorithm
        self.algorithm = ALGORITHM_TYPE.get(self.algorithm_config.algorithm_type)
        if self.algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(self.algorithm_config.advantage_fn)(
                **self.algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(self.algorithm_config.kl_penalty_fn)(
                **self.algorithm_config.kl_penalty_fn_args
            )

        # Training steps config
        self.total_training_steps = self.config.trainer.total_training_steps or sys.maxsize
        if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim") is not None:
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                self.total_training_steps
            )
        if OmegaConf.select(self.config, "critic.optim") is not None:
            self.config.critic.optim.total_training_steps = self.total_training_steps

        self.device_name = self.config.trainer.get("device", "cuda")
        self._init_workers()

    def _init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # Actor+Rollout
        actor_role = Role.ActorRollout
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            role=str(actor_role),
        )
        resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls

        # Critic (engine-based TrainingWorker)
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=critic_worker_cfg
            )
            resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # Spawn worker groups
        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config, "global_profiler.steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")

        for pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Initialize critic
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        # Initialize actor (last, for better KV cache memory estimation)
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # Reference policy
        if self.use_reference_policy:
            self.ref_policy_wg = self.actor_rollout_wg

    # ------------------------------------------------------------------
    # TrainEngineWrapper interface
    # ------------------------------------------------------------------

    @property
    def train_step_num(self) -> int:
        return self.global_steps

    async def prepare(self):
        self.actor_rollout_wg.set_algorithm(self.algorithm_config)
        self.global_steps = 0
        self._load_checkpoint()

    async def train_step(self, batch_exps: List[Experience]) -> Dict:  # noqa: C901
        batch = to_data_proto(
            batch_exps, self.tokenizer.pad_token_id, self.empty_model, self.logger
        )
        metrics = {}
        self.global_steps += 1
        timing_raw = {}

        with marked_timer("step", timing_raw):
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            if self.algorithm.can_balance_batch and self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            # Rollout correction
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                "bypass_mode", False
            )
            if bypass_recomputing_logprobs:
                if "rollout_log_probs" in batch.batch:
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
            else:
                if (batch.meta_info["model_versions"] != self.global_steps - 1).any():
                    self.logger.warning(
                        f"model_versions mismatch: {batch.meta_info['model_versions']} vs {self.global_steps - 1}"
                    )
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    metrics.update(
                        {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                    )
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                    if "rollout_log_probs" in batch.batch.keys():
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

            if self.algorithm.use_reference:
                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.algorithm.use_critic:
                with marked_timer("values", timing_raw):
                    values_dp = self._compute_values(batch)
                    batch = batch.union(values_dp)

            if self.algorithm.compute_advantage_in_trainer:
                with marked_timer("adv", timing_raw):
                    batch, kl_metrics = self.kl_fn.apply_kl_penalty_to_reward(batch)
                    metrics.update(prefix_metrics(kl_metrics, prefix="critic"))
                    batch, _ = self.advantage_fn(batch)
            else:
                if "token_level_scores" in batch.batch.keys():
                    assert "token_level_rewards" not in batch.batch.keys()
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Rollout correction (decoupled mode)
            if (
                rollout_corr_config is not None
                and "rollout_log_probs" in batch.batch
                and not bypass_recomputing_logprobs
            ):
                from verl.trainer.ppo.rollout_corr_helper import (
                    compute_rollout_correction_and_add_to_batch,
                )

                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                    batch, rollout_corr_config
                )
                metrics.update(is_metrics)

            # Update critic
            if self.algorithm.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self._update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # Update actor
            if (
                not self.algorithm.use_critic
                or self.config.trainer.critic_warmup <= self.global_steps
            ):
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self._update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

        # Collect metrics
        metrics.update(compute_data_metrics(batch=batch))
        timing_metrics = compute_timing_metrics(batch=batch, timing_raw=timing_raw)
        metrics.update({k.replace("timing_s/", "time/"): v for k, v in timing_metrics.items()})
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

        return metrics

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        os.makedirs(local_global_step_folder, exist_ok=True)
        with open(os.path.join(local_global_step_folder, ".full_checkpoint"), "w") as f:
            f.write("")

        actor_local_path = os.path.join(local_global_step_folder, "actor")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None)

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            global_step=self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_local_path,
                global_step=self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        await self.checkpoint_monitor.monitor_step.remote(self.global_steps)

    def sync_weight(self) -> None:
        self.actor_rollout_wg.update_weights(global_steps=self.global_steps)

    async def upload_state_dict(self):
        self.actor_rollout_wg.update_weights(global_steps=self.global_steps)

    async def save_state_dict(self):
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "actor"
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, global_step=self.global_steps)
        await self.checkpoint_monitor.monitor_step.remote(self.global_steps, is_state_dict=True)

    # ------------------------------------------------------------------
    # Internal training methods
    # ------------------------------------------------------------------

    def _get_dp_size(self) -> int:
        """Get data parallel size from actor worker group."""
        role = "actor"
        wg = self.actor_rollout_wg
        if role not in wg._dispatch_info:
            dp_rank_mapping = wg._query_dispatch_info(role)
            wg._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = wg._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics: dict):
        """Reorder batch so each DP rank gets similar total tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size()

        global_partition_lst = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=dp_size, equal_size=True
        )
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(),
            partitions=global_partition_lst,
            prefix="global_seqlen",
        )
        metrics.update(global_balance_stats)

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        return DataProto.from_tensordict(ref_log_prob)

    def _compute_old_log_prob(self, batch: DataProto):
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get(
            "calculate_sum_pi_squared", False
        )
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics").get("mfu", 0.0)
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        return DataProto.from_tensordict(values)

    def _update_actor(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.algorithm_config.entropy_loss_fn != "none"
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if "actor/mfu" in actor_output:
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )
        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        if "critic/mfu" in output:
            output["perf/mfu/critic"] = output.pop("critic/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": output})

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                self.logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str)
            assert "global_step_" in self.config.trainer.resume_from_path
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        self.logger.info(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        self.logger.info(f"Setting global step to {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            critic_path = os.path.join(global_step_folder, "critic")
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
