"""GRPO advantage computation

Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, List, Tuple

import torch

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.common.experience import Experience


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
        exps: List[Experience],
        **kwargs,
    ) -> Tuple[List[Experience], Dict]:
        """
        Compute advantage for GRPO. This method should only be called with experiences that belong to the same task.

        Args:
            exps (List[Experience]): List of experiences belonging to the same task. Contains at least 1 experience.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                group_reward_mean = torch.mean([exp.reward for exp in exps])
                group_reward_std = torch.std([exp.reward for exp in exps])
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }
