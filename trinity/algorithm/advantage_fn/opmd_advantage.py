"""OPMD advantage computation"""

from typing import Dict, List, Tuple

import torch

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.common.experience import Experience


@ADVANTAGE_FN.register_module("opmd")
class OPMDAdvantageFn(AdvantageFn):
    """OPMD advantage computation"""

    def __init__(
        self,
        opmd_baseline: str = "mean",
        tau: float = 1.0,
    ) -> None:
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def __call__(
        self,
        exps: List[Experience],
        **kwargs,
    ) -> Tuple[List[Experience], Dict]:
        """Modified from compute_grpo_outcome_advantage

        Compute advantage for OPMD, operating only on Outcome reward
        (with only one scalar reward for each response).

        Args:
            exps (List[Experience]): List of experiences belonging to the same task. Contains at least 1 experience.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                group_rewards = torch.tensor([exp.reward for exp in exps])
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                elif self.opmd_baseline == "logavgexp":
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
                else:
                    raise NotImplementedError(f"Unknown OPMD baseline: {self.opmd_baseline}")
                for exp in exps:
                    score = exp.reward - group_baseline
                    exp.advantages = score * exp.action_mask
                    exp.returns = exp.advantages
                metrics = {
                    "group_baseline": group_baseline,
                }
            return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "opmd_baseline": "mean",
            "tau": 1.0,
        }
