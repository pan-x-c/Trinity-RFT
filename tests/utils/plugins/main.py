from tests.utils.plugins.dependencies import DEPENDENCY_VALUE, dependency_func
from trinity.common.workflows.workflow import RepeatableWorkflow


class MainDummyWorkflow(RepeatableWorkflow):
    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

    def run(self) -> list:
        return [DEPENDENCY_VALUE, dependency_func()]
