import unittest
from unittest.mock import AsyncMock, MagicMock

import torch

from trinity.algorithm import ADD_STRATEGY
from trinity.common.experience import EID, Experience


class TestAddStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_grpo_args(self):
        writer = MagicMock()
        writer.write_async = AsyncMock()
        strategy = ADD_STRATEGY.get("grpo")(writer, epsilon=1e-7)
        self.assertEqual(strategy.epsilon, 1e-7)
        task_num = 3
        repeat_times = 5
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
        count, metrics = await strategy.add(exps, step=0)
        self.assertEqual(count, task_num * repeat_times)
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
        count, metrics = await strategy.add(exps, step=0)
        self.assertEqual(count, task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 0.0)
        self.assertTrue(metrics["group_advantages/reward_std/mean"] == 1.0)
