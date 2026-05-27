"""Writer of the SQL buffer."""

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.sql import SQLStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class SQLWriter(BufferWriter):
    """Writer of the SQL buffer."""

    def __init__(self, config: StorageConfig) -> None:
        assert config.storage_type == StorageType.SQL.value
        self.wrap_in_ray = config.wrap_in_ray
        self._db_wrapper = None
        self._async_storage = None
        self._config = config

    @property
    def db_wrapper(self):
        if self._db_wrapper is None:
            self._db_wrapper = SQLStorage.get_wrapper(self._config)
        return self._db_wrapper

    async def _get_async_storage(self):
        if self._async_storage is None:
            from trinity.buffer.storage.async_sql import (
                AsyncSQLExperienceStorage,
                AsyncSQLTaskStorage,
            )

            if self._config.schema_type is None:
                self._async_storage = AsyncSQLTaskStorage(self._config)
            else:
                self._async_storage = AsyncSQLExperienceStorage(self._config)
            await self._async_storage.init()
        return self._async_storage

    def write(self, data: list) -> None:
        if self.wrap_in_ray:
            ray.get(self.db_wrapper.write.remote(data))
        else:
            self.db_wrapper.write(data)

    async def write_async(self, data):
        if self.wrap_in_ray:
            await self.db_wrapper.write.remote(data)
        else:
            storage = await self._get_async_storage()
            await storage.write(data)

    async def acquire(self) -> int:
        if self.wrap_in_ray:
            return await self.db_wrapper.acquire.remote()
        else:
            storage = await self._get_async_storage()
            return storage.acquire()

    async def release(self) -> int:
        if self.wrap_in_ray:
            return await self.db_wrapper.release.remote()
        else:
            storage = await self._get_async_storage()
            return storage.release()
