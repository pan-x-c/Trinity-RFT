from typing import List, Tuple

import numpy as np

from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.experience import Experience, group_by


@EXPERIENCE_OPERATORS.register_module("reward_std_filter")
class RewardSTDFilter(ExperienceOperator):
    """
    Filter experiences based on the standard deviation of rewards within each group.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward std."""
        metrics = {}
        result_exps = []
        original_count = len(exps)
        grouped_experiences = group_by(exps, id_type="task")
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            rewards = [exp.reward for exp in group_exps]
            variance = np.std(rewards)
            if variance <= self.threshold:
                continue
            result_exps.extend(group_exps)
        final_count = len(result_exps)
        metrics["filtered_count"] = original_count - final_count
        return result_exps, metrics
