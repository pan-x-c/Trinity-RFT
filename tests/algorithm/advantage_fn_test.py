import unittest

import torch

from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.common.experience import EID, Experience


class TestGroupedAdvantageFn(unittest.TestCase):
    def test_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(**advantage_fn_cls.default_args())
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(batch=0, task=j, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        grouped_exps = advantage_fn.group_experiences(exps)
        self.assertEqual(len(grouped_exps), task_num)

        for group_id, group_exps in grouped_exps.items():
            modified_exps, group_metrics = advantage_fn.calculate_group_advantage(
                group_id, group_exps
            )
            self.assertEqual(len(modified_exps), repeat_times)
            self.assertIn("reward_mean", group_metrics)
            self.assertIn("reward_std", group_metrics)
