from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from trinity.common.experience import Experience
from trinity.data.operators import ExperienceOperator
from trinity.utils.monitor import gather_metrics
from trinity.utils.registry import Registry

ADVANTAGE_FN = Registry("advantage_fn")


class AdvantageFn(ABC):
    @abstractmethod
    def __call__(self, exps: Any, **kwargs: Dict) -> Tuple[Any, Dict]:
        """Calculate advantages from experiences

        Args:
            exps (`DataProto`): The input experiences.
            kwargs (`Dict`): The step-level parameters for calculating advantages.

        Returns:
            `DataProto`: The experiences with advantages.
            `Dict`: The metrics for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> Dict:
        """
        Returns:
            `Dict`: The default init arguments for the advantage function.
        """


class GroupAdvantage(ExperienceOperator):
    """For group-based advantages calculation."""

    @abstractmethod
    def group_experiences(self, exps: List[Experience]) -> Dict[str, List[Experience]]:
        """Group experiences by a certain criterion.

        Args:
            exps (List[Experience]): List of experiences to be grouped.

        Returns:
            Dict[str, List[Experience]]: A dictionary where keys are group identifiers and values are lists of experiences.
        """

    @abstractmethod
    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        """Calculate advantages for a group of experiences.

        Args:
            group_id (str): The identifier for the group of experiences.
            exps (List[Experience]): List of experiences in the group.

        Returns:
            List[Experience]: A tuple containing the modified list of experiences and a dictionary of metrics.
        """

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}
        exp_groups = self.group_experiences(exps)
        cnt = 0
        metric_list = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
        try:
            metrics = gather_metrics(metric_list, "group_advantages")
            metrics["experience_count"] = cnt
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        exps = [exp for group in exp_groups.values() for exp in group]  # Flatten the list
        return exps, metrics
