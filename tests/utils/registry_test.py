import unittest

from trinity.common.workflows import WORKFLOWS, Workflow


class TestRegistry(unittest.TestCase):
    def test_dynamic_import(self):
        workflow_cls = WORKFLOWS.get("tests.utils.plugins.main.MainDummyWorkflow")
        self.assertTrue(issubclass(workflow_cls, Workflow))
        workflow = workflow_cls(task=None, model=None)
        res = workflow.run()
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], "0")
