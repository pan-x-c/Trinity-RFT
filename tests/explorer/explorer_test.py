"""Tests for explorer."""
import os
import unittest
from datetime import datetime

import ray

from tests.tools import (
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli.launcher import explore
from trinity.common.constants import MonitorType


class BaseExplorerCase:
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm_async"
        self.config.explorer.repeat_times = 2
        self.config.monitor.monitor_type = MonitorType.TENSORBOARD
        self.config.monitor.project = "Trinity-unittest"
        self.config.model.checkpoint_path = get_checkpoint_path()
        self.config.synchronizer.sync_iteration_interval = 2
        self.config.explorer.eval_interval = 4
        self.config.trainer.eval_interval = 4


class TestExplorerCountdownEval(BaseExplorerCase, unittest.TestCase):
    def test_explorer(self):
        self.config.data = get_unittest_dataset_config("countdown")
        self.config.monitor.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)

    def tearDown(self):
        pass


class TestExplorerCountdownNoEval(BaseExplorerCase, unittest.TestCase):
    def test_explorer(self):
        self.config.data = get_unittest_dataset_config("countdown")
        self.config.monitor.name = f"explore-no-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.data.eval_split = None
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)

    def tearDown(self):
        pass
