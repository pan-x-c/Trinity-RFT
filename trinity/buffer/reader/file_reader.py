"""Filed based buffer reader."""

from typing import List, Optional

import datasets
from datasets import Dataset, load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema.formatter import FORMATTER
from trinity.common.config import StorageConfig


class DummyProgressBar:
    def __init__(self):
        pass

    def update(self, num: int):
        pass

    def close(self):
        pass


class _HFBatchReader:
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        default_batch_size: int,
        total_epochs: int = 1,
        offset: int = 0,
        drop_last: bool = True,
        total_steps: Optional[int] = None,
        enable_progress_bar: Optional[bool] = True,
    ):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.name = name
        self.current_batch_size = None
        self.drop_last = drop_last

        self.current_offset = offset
        self.iter = iter(self.dataset)

        for _ in range(self.current_offset % self.dataset_size):
            next(self.iter)

        # convert epochs/steps to sample number
        if total_steps:
            self.total_samples = default_batch_size * total_steps
        else:
            self.total_samples = self.dataset_size * total_epochs

        if enable_progress_bar:
            from ray.experimental.tqdm_ray import tqdm

            self.progress_bar = tqdm(
                total=self.total_samples,
                desc=f"Dataset [{self.name}] Progressing",
            )
        else:
            self.progress_bar = DummyProgressBar()

        self.progress_bar.update(self.current_offset)

    def read_batch(self, batch_size: int) -> List:
        if self.current_offset >= self.total_samples:
            self.progress_bar.close()
            raise StopIteration
        batch = []

        while len(batch) < batch_size:
            try:
                item = next(self.iter)
                batch.append(item)
                self.current_offset += 1
            except StopIteration:
                if self.current_offset >= self.total_samples:
                    # No more data to read
                    if not self.drop_last and len(batch) > 0:
                        # return last batch
                        self.progress_bar.update(len(batch))
                        return batch
                    else:
                        self.progress_bar.close()
                        raise StopIteration
                # Step to the next epoch
                self.iter = iter(self.dataset)
        self.progress_bar.update(batch_size)
        return batch


class BaseFileReader(BufferReader):
    async def read_async(self, batch_size: Optional[int] = None):
        try:
            return self.read(batch_size)
        except StopIteration as e:
            raise StopAsyncIteration from e


class ExperienceFileReader(BaseFileReader):
    """Reader for SFT / DPO file data."""

    def __init__(self, config: StorageConfig):
        self.formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        self.read_batch_size = config.batch_size
        self.dataset = _HFBatchReader(
            load_dataset(config.path, name=config.subset_name, split=config.split),
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs,
            drop_last=True,
            total_steps=config.total_steps,
            enable_progress_bar=config.enable_progress_bar,
        )

    def read(self, batch_size: Optional[int] = None) -> List:
        samples = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in samples:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list


class TaskFileReader(BaseFileReader):
    """A Reader for task file data."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.name = config.name
        self.epoch = 0
        datasets.disable_caching()
        self.read_batch_size = config.batch_size
        self.dataset = _HFBatchReader(
            load_dataset(self.config.path, name=self.config.subset_name, split=self.config.split),
            name=self.config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=self.config.total_epochs if not self.config.is_eval else 1,
            offset=self.config.index,
            drop_last=not self.config.is_eval,
            total_steps=self.config.total_steps,
            enable_progress_bar=self.config.enable_progress_bar,
        )
        self.formatter = FORMATTER.get("task")(config)

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples = self.dataset.read_batch(batch_size)
        for sample in samples:
            task = self.formatter.format(sample)
            tasks.append(task)
        return tasks
