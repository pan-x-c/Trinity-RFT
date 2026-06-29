from abc import ABC, abstractmethod
from typing import List

from trinity.common.experience import Experience


class BaseStore(ABC):
    """Abstract base class for an in-process experience store.

    The key follows the format ``<batch_id>/<task_id>/<run_id>`` and each
    experience is associated with a unique sample id.
    """

    @abstractmethod
    def add(self, key: str, exps: List[Experience]) -> None:
        """Add experiences to the store under the given complete key."""

    @abstractmethod
    def overwrite(self, key: str, exps: List[Experience]) -> None:
        """Replace all experiences under the given complete key."""

    @abstractmethod
    def replace(self, key: str, old_sample_id: str, exp: Experience) -> None:
        """Replace one experience under the given complete key."""

    @abstractmethod
    def update(
        self, key: str, reward: float, info: dict | None, sample_ids: List[str] | None
    ) -> None:
        """Update reward, EID fields from key, and optional info for selected experiences."""

    @abstractmethod
    def get(self, key: str) -> List[Experience]:
        """Return experiences for an exact key or prefix without removing them."""

    @abstractmethod
    def remove(self, key: str) -> List[Experience]:
        """Remove and return experiences for an exact key or prefix."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Return complete keys currently stored in insertion order."""


RecordStore = BaseStore
