"""Harbor taskset reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema.formatter import TaskFormatter
from trinity.common.config import StorageConfig


def _ensure_harbor_importable() -> None:
    try:
        import harbor  # noqa: F401

        return
    except ImportError as exc:
        raise ImportError(
            "HarborReader requires the `harbor` package to be installed in the "
            "current Python environment."
        ) from exc


@dataclass(frozen=True)
class HarborTaskEntry:
    index: int
    task_id: str
    harbor_task_name: str
    harbor_task_short_name: str
    harbor_task_path: str
    harbor_dataset_name: str
    harbor_source_type: str = "local"
    harbor_has_steps: bool = False
    harbor_environment_os: str = "linux"
    harbor_metadata: dict | None = None

    def to_sample(self) -> dict:
        return {
            "task_id": self.task_id,
            "harbor_task_name": self.harbor_task_name,
            "harbor_task_short_name": self.harbor_task_short_name,
            "harbor_task_path": self.harbor_task_path,
            "harbor_dataset_name": self.harbor_dataset_name,
            "harbor_source_type": self.harbor_source_type,
            "harbor_has_steps": self.harbor_has_steps,
            "harbor_environment_os": self.harbor_environment_os,
            "harbor_metadata": self.harbor_metadata or {},
        }


class _HarborTaskBatchReader:
    def __init__(
        self,
        entries: list[HarborTaskEntry],
        name: str,
        default_batch_size: int,
        total_epochs: int = 1,
        offset: int = 0,
        drop_last: bool = True,
        total_steps: Optional[int] = None,
    ):
        self.entries = entries
        self.dataset_size = len(entries)
        if self.dataset_size == 0:
            raise ValueError(f"Harbor dataset [{name}] is empty and cannot be read.")
        self.name = name
        self.default_batch_size = default_batch_size
        self.drop_last = drop_last
        self.current_offset = offset

        if total_steps is not None:
            self.total_samples = default_batch_size * total_steps
        else:
            self.total_samples = self.dataset_size * total_epochs

    def read_batch(self, batch_size: int) -> Tuple[List[dict], List[int]]:
        batch, indices = [], []
        while len(batch) < batch_size:
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                raise StopIteration

            index = self.current_offset % self.dataset_size
            batch.append(self.entries[index].to_sample())
            indices.append(index)
            self.current_offset += 1

        return batch, indices

    def select_batch(self, indices: List[int]) -> List[dict]:
        batch = []
        for i in indices:
            if not 0 <= i < self.dataset_size:
                raise IndexError(f"Harbor task index {i} out of range.")
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                raise StopIteration
            batch.append(self.entries[int(i)].to_sample())
            self.current_offset += 1
        return batch


class HarborReader(BufferReader):
    """Read local Harbor task directories as Trinity workflow tasks.

    The reader keeps Harbor task directories as the source of truth. Each
    Trinity raw task is an index record pointing at a Harbor task directory.
    """

    def __init__(self, config: StorageConfig):
        _ensure_harbor_importable()

        self.config = config
        self.name = config.name
        self.read_batch_size = config.batch_size
        self.formatter = TaskFormatter(config)
        self.entries = self._discover_local_tasks(config)
        self.dataset = _HarborTaskBatchReader(
            self.entries,
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs if not config.is_eval else 1,
            offset=config.index,
            drop_last=not config.is_eval,
            total_steps=config.total_steps if not config.is_eval else None,
        )
        self._init_selector(config)

    def _init_selector(self, config: StorageConfig) -> None:
        if config.data_selector is not None:
            from trinity.buffer.selector import SELECTORS
            from trinity.buffer.selector.selector import BaseSelector

            selector_cls = SELECTORS.get(config.data_selector.selector_type)
            self.selector: BaseSelector = selector_cls(self.dataset, config.data_selector)
        else:
            self.selector = None

    def _discover_local_tasks(self, config: StorageConfig) -> list[HarborTaskEntry]:
        if config.path is None:
            raise ValueError("HarborReader requires `path` to be configured.")

        from harbor.models.task.task import Task as HarborTask

        root = Path(config.path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Harbor taskset path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Harbor taskset path must be a directory: {root}")

        disable_verification = config.workflow_args.get("harbor_disable_task_verification", False)
        candidate_dirs = (
            [root]
            if HarborTask.is_valid_dir(root, disable_verification=disable_verification)
            else sorted(root.iterdir())
        )
        entries: list[HarborTaskEntry] = []

        for candidate in candidate_dirs:
            if not candidate.is_dir():
                continue
            if not HarborTask.is_valid_dir(
                candidate,
                disable_verification=disable_verification,
            ):
                continue
            harbor_task = HarborTask(candidate)
            harbor_task_name = harbor_task.name
            task_id = f"{config.name}:{len(entries)}:{harbor_task_name}"
            entries.append(
                HarborTaskEntry(
                    index=len(entries),
                    task_id=task_id,
                    harbor_task_name=harbor_task_name,
                    harbor_task_short_name=harbor_task.short_name,
                    harbor_task_path=str(harbor_task.task_dir),
                    harbor_dataset_name=config.name,
                    harbor_has_steps=harbor_task.has_steps,
                    harbor_environment_os=harbor_task.config.environment.os.value,
                    harbor_metadata=dict(harbor_task.config.metadata),
                )
            )

        return entries

    async def read(self, batch_size: Optional[int] = None, **kwargs):
        try:
            return self._read_sync(batch_size, **kwargs)
        except StopIteration as e:
            raise StopAsyncIteration from e

    def _read_sync(self, batch_size: Optional[int] = None, **kwargs):
        batch_size = batch_size or self.read_batch_size
        if self.selector is not None:
            indices = self.selector.get_indices(batch_size)
            samples = self.dataset.select_batch(indices)
        else:
            samples, indices = self.dataset.read_batch(batch_size)

        tasks = []
        for sample, index in zip(samples, indices):
            task = self.formatter.format(sample)
            task.index["index"] = int(index)
            task.index["harbor_task_name"] = sample["harbor_task_name"]
            task.index["harbor_task_path"] = sample["harbor_task_path"]
            tasks.append(task)
        return tasks

    def state_dict(self):
        if self.selector is not None:
            return self.selector.state_dict()
        return {"current_index": self.dataset.current_offset}

    def load_state_dict(self, state_dict):
        if self.selector is not None:
            self.selector.load_state_dict(state_dict)
        else:
            self.dataset.current_offset = state_dict["current_index"]

    def feedback(self, **pipeline_metrics):
        if self.selector is not None:
            self.selector.feedback(**pipeline_metrics)

    def __len__(self):
        return self.dataset.dataset_size
