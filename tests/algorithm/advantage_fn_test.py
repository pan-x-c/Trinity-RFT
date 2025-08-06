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

        # test group_epxeriences
        grouped_exps = advantage_fn.group_experiences(exps)
        self.assertEqual(len(grouped_exps), task_num)

        # test calculate_group_advantage
        for group_id, group_exps in grouped_exps.items():
            modified_exps, group_metrics = advantage_fn.calculate_group_advantage(
                group_id, group_exps
            )
            self.assertEqual(len(modified_exps), repeat_times)
            self.assertIn("reward_mean", group_metrics)
            self.assertIn("reward_std", group_metrics)

        # test the full pipeline

        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 2.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )

        repeat_times = 1
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
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 0.0)
        self.assertTrue(metrics["group_advantages/reward_std/mean"] == 1.0)


    def test_step_wise_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("step_wise_grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7)
        self.assertEqual(advantage_fn.epsilon, 1e-7)
        task_num = 2
        repeat_times = 3
        step_num = 4
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                    step=k,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for k in range(step_num)
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times * step_num)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 1.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )
