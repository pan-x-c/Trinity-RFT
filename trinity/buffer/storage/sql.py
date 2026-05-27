"""SQL database storage — sync wrappers around async implementations."""

import asyncio
from typing import Dict, List, Optional

import ray
from datasets import Dataset
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from trinity.buffer.storage.async_sql import (
    AsyncSQLExperienceStorage,
    AsyncSQLTaskStorage,
)
from trinity.common.config import StorageConfig
from trinity.common.experience import Experience
from trinity.common.workflows import Task


class _SyncBridge:
    """Mixin providing a dedicated event loop for sync-to-async bridging."""

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _init_loop(self):
        self._loop = asyncio.new_event_loop()


class SQLExperienceStorage(_SyncBridge):
    """Sync wrapper around AsyncSQLExperienceStorage for offline tools (viewer, benchmarks)."""

    def __init__(self, config: StorageConfig) -> None:
        self._init_loop()
        self._async = AsyncSQLExperienceStorage(config)
        self._run(self._async.init())
        self.engine = create_engine(config.path, poolclass=NullPool)
        self.table_model_cls = self._async.table_model_cls
        self.blob_model_cls = self._async.blob_model_cls
        self.batch_size = self._async.batch_size

    @property
    def max_experience_bytes(self):
        return self._async.max_experience_bytes

    @max_experience_bytes.setter
    def max_experience_bytes(self, value):
        self._async.max_experience_bytes = value

    def write(self, data: List[Experience]) -> None:
        self._run(self._async.write(data))

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List[Experience]:
        try:
            return self._run(self._async.read(batch_size, **kwargs))
        except StopAsyncIteration:
            raise StopIteration()

    def count(self, filters: Optional[Dict] = None) -> int:
        return self._run(self._async.count(filters))

    def query(
        self, offset: int = 0, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Experience]:
        return self._run(self._async.query(offset, limit, filters))

    def acquire(self) -> int:
        return self._async.acquire()

    def release(self) -> int:
        return self._async.release()

    @classmethod
    def load_from_dataset(cls, dataset: Dataset, config: StorageConfig) -> "SQLExperienceStorage":
        storage = cls(config)
        loop = storage._loop
        async_storage = storage._async
        from trinity.buffer.schema import FORMATTER

        formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        batch_size = storage.batch_size
        batch = []
        for item in dataset:
            batch.append(formatter.format(item))
            if len(batch) >= batch_size:
                loop.run_until_complete(async_storage.write(batch))
                batch.clear()
        if batch:
            loop.run_until_complete(async_storage.write(batch))
        return storage


class SQLTaskStorage(_SyncBridge):
    """Sync wrapper around AsyncSQLTaskStorage for offline tools."""

    def __init__(self, config: StorageConfig) -> None:
        self._init_loop()
        self._async = AsyncSQLTaskStorage(config)
        self._run(self._async.init())
        self.engine = create_engine(config.path, poolclass=NullPool)
        self.table_model_cls = self._async.table_model_cls
        self.batch_size = self._async.batch_size

    def write(self, data: List[Dict]) -> None:
        self._run(self._async.write(data))

    def read(self, batch_size: Optional[int] = None) -> List[Task]:
        try:
            return self._run(self._async.read(batch_size))
        except StopAsyncIteration:
            raise StopIteration()

    def acquire(self) -> int:
        return self._async.acquire()

    def release(self) -> int:
        return self._async.release()

    @classmethod
    def load_from_dataset(cls, dataset: Dataset, config: StorageConfig) -> "SQLTaskStorage":
        storage = cls(config)
        loop = storage._loop
        async_storage = storage._async
        batch_size = config.batch_size
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= batch_size:
                loop.run_until_complete(async_storage.write(batch))
                batch.clear()
        if batch:
            loop.run_until_complete(async_storage.write(batch))
        return storage


class SQLStorage:
    """Factory for creating SQL storage instances, optionally wrapped in Ray actors."""

    @classmethod
    def get_wrapper(cls, config: StorageConfig):
        if config.schema_type is None:
            sync_cls = SQLTaskStorage
            async_cls = AsyncSQLTaskStorage
        else:
            sync_cls = SQLExperienceStorage
            async_cls = AsyncSQLExperienceStorage
        if config.wrap_in_ray:
            return (
                ray.remote(async_cls)
                .options(
                    name=f"sql-{config.name}",
                    namespace=config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                    max_concurrency=5,
                )
                .remote(config)
            )
        else:
            return sync_cls(config)
