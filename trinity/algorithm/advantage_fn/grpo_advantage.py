"""GRPO advantage computation
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import (
    ADVANTAGE_FN,
    AdvantageFn,
    GroupAdvantage,
)
from trinity.common.experience import Experience, group_by
from trinity.data.operators import EXPERIENCE_OPERATORS


@ADVANTAGE_FN.register_module("grpo")
class GRPOAdvantageFn(AdvantageFn):
    """GRPO advantage computation"""

    def __init__(
        self,
        epsilon: float = 1e-6,
    ) -> None:
        self.epsilon = epsilon

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        Compute advantage for GRPO, operating only on Outcome reward
        (with only one scalar reward for each response).
        Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            scores: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        eos_mask = exps.batch["response_mask"]
        index = exps.non_tensor_batch["uid"]
        epsilon = self.epsilon

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }


@EXPERIENCE_OPERATORS.register_module("grpo")
class GRPOGroupedAdvantage(GroupAdvantage):
    """An example AddStrategy that calculates GRPO advantages."""

    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@EXPERIENCE_OPERATORS.register_module("correct_bias_grpo")
class CorrectBiasMapper(GRPOGroupedAdvantage):
    """An Addstrategy with GroupAdvantage that corrects for rank bias (https://arxiv.org/pdf/2506.02355)"""

    def __init__(self, epsilon: float = 1e-6, rank_penalty: float = 0.25, **kwargs) -> None:
        super().__init__(epsilon)
        self.rank_penalty = rank_penalty

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                # correct bias
                old_log_probs = torch.tensor([torch.mean(exp.logprobs, axis=-1) for exp in exps])
                group_ranks = torch.argsort(torch.argsort(old_log_probs))
                group_ranks = group_ranks / len(group_ranks)
                rewards = rewards * (1 - group_ranks * self.rank_penalty)

                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            for i, exp in enumerate(exps):
                score = (rewards[i] - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6, "rank_penalty": 0.25}
