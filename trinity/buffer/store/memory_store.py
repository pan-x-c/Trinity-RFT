"""In-memory implementation of the experience store interface."""

from collections import OrderedDict
from typing import Callable, Iterable, List

from trinity.buffer.store.base_store import BaseStore
from trinity.common.experience import Experience

SampleIdGetter = Callable[[Experience], str]


def default_sample_id_getter(exp: Experience) -> str:
    """Resolve a stable sample id for an experience."""
    info = exp.info or {}
    sample_id = info.get("sample_id")
    if sample_id is not None:
        return str(sample_id)

    request_id = info.get("request_id")
    sample_index = info.get("sample_index")
    if request_id is not None and sample_index is not None:
        return f"{request_id}:{sample_index}"
    if request_id is not None:
        return str(request_id)

    return exp.eid.uid


class MemoryStore(BaseStore):
    """A fast in-process store backed by Python dictionaries.

    ``add``, ``overwrite`` and ``update`` require complete keys in the form
    ``<step_id>/<task_id>/<run_id>``. ``get`` and ``remove`` also accept prefixes
    so callers can drain a task or step at once.
    """

    def __init__(self, sample_id_getter: SampleIdGetter | None = None) -> None:
        self.sample_id_getter = sample_id_getter or default_sample_id_getter
        self._records: dict[str, OrderedDict[str, Experience]] = {}
        self._sample_to_key: dict[str, str] = {}

    def __len__(self) -> int:
        return sum(len(exps) for exps in self._records.values())

    def add(self, key: str, exps: List[Experience]) -> None:
        self._validate_complete_key(key)
        if not exps:
            return

        records = self._records.setdefault(key, OrderedDict())
        for exp in exps:
            sample_id = self.sample_id_getter(exp)
            owner_key = self._sample_to_key.get(sample_id)
            if owner_key is not None:
                raise ValueError(
                    f"Duplicate sample_id '{sample_id}' already exists under key '{owner_key}'."
                )
            records[sample_id] = exp
            self._sample_to_key[sample_id] = key

    def overwrite(self, key: str, exps: List[Experience]) -> None:
        self._validate_complete_key(key)
        self._drop_key(key)
        self.add(key, exps)

    def update(
        self,
        key: str,
        reward: float,
        info: dict | None,
        sample_ids: List[str] | None,
    ) -> None:
        self._validate_complete_key(key)
        records = self._records.get(key)
        if records is None:
            raise KeyError(f"Key '{key}' does not exist.")

        target_ids: Iterable[str] = list(records.keys()) if sample_ids is None else sample_ids
        for sample_id in target_ids:
            if sample_id not in records:
                raise KeyError(f"sample_id '{sample_id}' does not exist under key '{key}'.")
            exp = records[sample_id]
            exp.reward = reward
            if info:
                if exp.info is None:
                    exp.info = {}
                exp.info.update(info)

    def get(self, key: str) -> List[Experience]:
        result: List[Experience] = []
        for matched_key in self._matching_keys(key):
            result.extend(self._records[matched_key].values())
        return result

    def remove(self, key: str) -> List[Experience]:
        result: List[Experience] = []
        for matched_key in self._matching_keys(key):
            result.extend(self._drop_key(matched_key))
        return result

    def keys(self) -> list[str]:
        return list(self._records.keys())

    @staticmethod
    def _validate_complete_key(key: str) -> None:
        parts = key.split("/")
        if len(parts) != 3 or any(part == "" for part in parts):
            raise ValueError(
                f"Store key must be complete '<step_id>/<task_id>/<run_id>', got '{key}'."
            )

    def _matching_keys(self, key: str) -> list[str]:
        if key == "":
            return list(self._records.keys())
        if key in self._records:
            return [key]
        prefix = key + "/"
        return [record_key for record_key in self._records if record_key.startswith(prefix)]

    def _drop_key(self, key: str) -> list[Experience]:
        records = self._records.pop(key, None)
        if records is None:
            return []
        for sample_id in records:
            self._sample_to_key.pop(sample_id, None)
        return list(records.values())
