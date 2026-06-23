import shutil
import unittest
from pathlib import Path

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.reader.harbor_reader import HarborReader
from trinity.common.config import TasksetConfig
from trinity.common.constants import StorageType


class TestHarborReader(unittest.IsolatedAsyncioTestCase):
    temp_dir = Path("tmp/test_harbor_reader")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        cls.temp_dir.mkdir(parents=True)
        cls._write_task("task-a", "harbor/task-a", "Create a.txt.")
        cls._write_task("task-b", "harbor/task-b", "Create b.txt.")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _write_task(cls, dirname: str, task_name: str, instruction: str) -> None:
        task_dir = cls.temp_dir / dirname
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "solution").mkdir()
        (task_dir / "task.toml").write_text(
            "\n".join(
                [
                    'version = "1.0"',
                    "",
                    "[task]",
                    f'name = "{task_name}"',
                    "",
                    "[environment]",
                    "cpus = 1",
                    "",
                    "[agent]",
                    "timeout_sec = 120.0",
                    "",
                    "[verifier]",
                    "timeout_sec = 120.0",
                    "",
                ]
            )
        )
        (task_dir / "instruction.md").write_text(instruction)
        (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.11-slim\n")
        (task_dir / "tests" / "test.sh").write_text(
            "#!/bin/bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n"
        )
        (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")

    def _config(self, *, index: int = 0, total_epochs: int = 1) -> TasksetConfig:
        config = TasksetConfig(
            name="harbor_test",
            storage_type=StorageType.HARBOR.value,
            path=str(self.temp_dir),
            default_workflow_type="simple_workflow",
            batch_size=1,
            index=index,
            total_epochs=total_epochs,
        )
        config.data_selector = None
        return config

    async def test_read_harbor_tasks(self):
        reader = get_buffer_reader(self._config())

        self.assertIsInstance(reader, HarborReader)
        self.assertEqual(len(reader), 2)

        tasks = await reader.read(batch_size=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].raw_task["harbor_task_name"], "harbor/task-a")
        self.assertEqual(tasks[0].raw_task["harbor_task_short_name"], "task-a")
        self.assertEqual(tasks[0].raw_task["harbor_dataset_name"], "harbor_test")
        self.assertEqual(tasks[0].raw_task["harbor_source_type"], "local")
        self.assertTrue(Path(tasks[0].raw_task["harbor_task_path"]).is_absolute())
        self.assertEqual(tasks[0].index["index"], 0)
        self.assertEqual(tasks[1].index["index"], 1)

    async def test_resume_offset(self):
        reader = get_buffer_reader(self._config(index=1, total_epochs=2))
        tasks = await reader.read(batch_size=2)

        self.assertEqual(
            [task.raw_task["harbor_task_name"] for task in tasks],
            [
                "harbor/task-b",
                "harbor/task-a",
            ],
        )
        self.assertEqual(reader.state_dict()["current_index"], 3)


if __name__ == "__main__":
    unittest.main()
