"""Tests for the async SQL buffer implementation."""

import os
import unittest

import torch

from trinity.buffer.storage.async_sql import (
    AsyncSQLExperienceStorage,
    AsyncSQLTaskStorage,
)
from trinity.buffer.utils import to_async_url
from trinity.common.config import StorageConfig
from trinity.common.experience import Experience

db_path = os.path.join(os.path.dirname(__file__), "test_async.db")


class TestToAsyncUrl(unittest.TestCase):
    """Tests for URL dialect conversion."""

    def test_sqlite(self):
        self.assertEqual(
            to_async_url("sqlite:///path/to/db.sqlite"),
            "sqlite+aiosqlite:///path/to/db.sqlite",
        )

    def test_postgresql(self):
        self.assertEqual(
            to_async_url("postgresql://user:pass@host/db"),
            "postgresql+asyncpg://user:pass@host/db",
        )

    def test_mysql(self):
        self.assertEqual(
            to_async_url("mysql://user:pass@host/db"),
            "mysql+aiomysql://user:pass@host/db",
        )

    def test_already_async(self):
        url = "sqlite+aiosqlite:///path/to/db.sqlite"
        self.assertEqual(to_async_url(url), url)


class TestAsyncSQLExperienceStorage(unittest.IsolatedAsyncioTestCase):
    """Tests for AsyncSQLExperienceStorage."""

    def _make_config(self, schema_type="experience") -> StorageConfig:
        config = StorageConfig()
        config.name = f"test_async_{schema_type}"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = schema_type
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 3
        return config

    def _make_experiences(self, num: int, token_length: int = 32) -> list:
        return [
            Experience(
                tokens=torch.randint(0, 1000, (token_length,), dtype=torch.int32),
                prompt_length=token_length // 4,
                reward=float(i) / num,
                logprobs=torch.randn(token_length - token_length // 4),
                info={"model_version": i % 3},
            )
            for i in range(num)
        ]

    async def test_write_and_read_fifo(self):
        """Write experiences and read them back in FIFO order."""
        config = self._make_config(schema_type="sft")
        storage = AsyncSQLExperienceStorage(config)
        await storage.init()

        exps = self._make_experiences(8)
        await storage.write(exps)

        result = await storage.read(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await storage.read(batch_size=4)
        self.assertEqual(len(result2), 4)

    async def test_write_and_read_priority(self):
        """Write experiences and read them back in priority order."""
        config = self._make_config(schema_type="experience")
        storage = AsyncSQLExperienceStorage(config)
        await storage.init()

        exps = self._make_experiences(8)
        await storage.write(exps)

        result = await storage.read(batch_size=8)
        self.assertEqual(len(result), 8)

    async def test_stop_iteration_on_release(self):
        """After release, read should raise StopAsyncIteration."""
        config = self._make_config(schema_type="sft")
        storage = AsyncSQLExperienceStorage(config)
        await storage.init()

        exps = self._make_experiences(4)
        await storage.write(exps)

        storage.acquire()
        storage.release()

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=4)

    async def test_oversized_experience_skipped(self):
        """Experiences exceeding max_experience_bytes should be skipped."""
        config = self._make_config(schema_type="sft")
        storage = AsyncSQLExperienceStorage(config)
        await storage.init()

        small_exps = self._make_experiences(2, token_length=16)
        large_exps = self._make_experiences(3, token_length=2048)

        small_size = len(small_exps[0].serialize())
        large_size = len(large_exps[0].serialize())
        storage.max_experience_bytes = (small_size + large_size) // 2

        await storage.write(large_exps + small_exps)

        result = await storage.read(batch_size=2)
        self.assertEqual(len(result), 2)

    async def test_read_timeout(self):
        """Read should raise StopAsyncIteration when timeout is reached."""
        config = self._make_config(schema_type="sft")
        config.max_read_timeout = 1
        storage = AsyncSQLExperienceStorage(config)
        await storage.init()

        exps = self._make_experiences(2)
        await storage.write(exps)

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=10)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


class TestAsyncSQLTaskStorage(unittest.IsolatedAsyncioTestCase):
    """Tests for AsyncSQLTaskStorage."""

    def _make_config(self) -> StorageConfig:
        config = StorageConfig()
        config.name = "test_async_task"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = None
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 5
        config.default_workflow_type = "math_workflow"
        return config

    async def test_write_and_read_tasks(self):
        """Write tasks and read them back."""
        config = self._make_config()
        storage = AsyncSQLTaskStorage(config)
        await storage.init()

        tasks = [{"question": f"q_{i}", "answer": f"a_{i}"} for i in range(8)]
        await storage.write(tasks)

        result = await storage.read(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await storage.read(batch_size=4)
        self.assertEqual(len(result2), 4)

    async def test_stop_iteration_on_empty(self):
        """Read should raise StopAsyncIteration when no data available."""
        config = self._make_config()
        storage = AsyncSQLTaskStorage(config)
        await storage.init()

        with self.assertRaises(StopAsyncIteration):
            await storage.read(batch_size=4)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


class TestAsyncReaderWriter(unittest.IsolatedAsyncioTestCase):
    """Tests for SQLReader/SQLWriter async methods with async storage."""

    def _make_config(self) -> StorageConfig:
        config = StorageConfig()
        config.name = "test_async_rw"
        config.path = f"sqlite:///{db_path}"
        config.schema_type = "sft"
        config.storage_type = "sql"
        config.wrap_in_ray = False
        config.batch_size = 4
        config.max_read_timeout = 3
        return config

    async def test_writer_reader_async_roundtrip(self):
        """Writer.write_async and Reader.read_async should work end-to-end."""
        from trinity.buffer.reader.sql_reader import SQLReader
        from trinity.buffer.writer.sql_writer import SQLWriter

        config = self._make_config()
        writer = SQLWriter(config)
        reader = SQLReader(config)

        await writer.acquire()

        exps = [
            Experience(
                tokens=torch.randint(0, 1000, (32,), dtype=torch.int32),
                prompt_length=8,
                reward=float(i),
                logprobs=torch.randn(24),
                info={"model_version": i},
            )
            for i in range(8)
        ]

        await writer.write_async(exps)

        result = await reader.read_async(batch_size=4)
        self.assertEqual(len(result), 4)

        result2 = await reader.read_async(batch_size=4)
        self.assertEqual(len(result2), 4)

        await writer.release()

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
