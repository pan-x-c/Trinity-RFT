import unittest
from pathlib import Path

from trinity.common.workflows import WORKFLOWS
from trinity.utils.plugin_loader import load_plugins


class TestPluginLoader(unittest.TestCase):
    def test_load_plugins(self):
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        self.assertIsNone(my_plugin_cls)
        load_plugins(Path(__file__).resolve().parent / "plugins")
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        self.assertIsNotNone(my_plugin_cls)
        my_plugin = my_plugin_cls(None, None, None)
        self.assertTrue(my_plugin.__module__.startswith("trinity.plugins"))
        res = my_plugin.run()
        self.assertEqual(res[0], "Hello world")
        self.assertEqual(res[1], "Hi")
