"""SQL database storage"""
import time
from abc import abstractmethod
from typing import Dict, List, Optional

import ray
from sqlalchemy import asc, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.schema import Base, create_dynamic_table
from trinity.buffer.schema.formatter import TaskFormatter
from trinity.buffer.utils import default_storage_path, retry_session
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.experience import Experience
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.log import get_logger


class SQLStorage:
    """
    An Storage based on SQL Database.

    If `wrap_in_ray` in `StorageConfig` is `True`, this class will be run as a Ray Actor,
    and provide a remote interface to the local database.

    For databases that do not support multi-processing read/write (e.g. sqlite, duckdb), please
    set `wrap_in_ray` to `True`.
    """

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(f"sql_{storage_config.name}")
        if storage_config.path is None:
            storage_config.path = default_storage_path(storage_config, config)
        self.engine = create_engine(storage_config.path, poolclass=NullPool)
        self.table_model_cls = create_dynamic_table(
            storage_config.name,
            storage_config.schema_type,
        )

        try:
            Base.metadata.create_all(self.engine, checkfirst=True)
        except OperationalError:
            self.logger.warning("Failed to create database, assuming it already exists.")

        self.session = sessionmaker(bind=self.engine)
        self.batch_size = config.train_batch_size
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval
        self.ref_count = 0
        self.stopped = False

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        if storage_config.schema_type is None:
            storage_cls = SQLExperienceStorage
        else:
            storage_cls = SQLExperienceStorage
        if storage_config.wrap_in_ray:
            return (
                ray.remote(storage_cls)
                .options(
                    name=f"sql-{storage_config.name}",
                    namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                )
                .remote(storage_config, config)
            )
        else:
            return storage_cls(storage_config, config)

    @abstractmethod
    def write(self, data: List) -> None:
        """Write a batch of data."""

    @abstractmethod
    def read(self, batch_size: Optional[int] = None) -> List:
        """Read a batch of data."""

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.stopped = True
        return self.ref_count


class SQLExperienceStorage(SQLStorage):
    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        super().__init__(storage_config, config)
        self.is_eval = storage_config.is_eval
        self.default_workflow_cls = WORKFLOWS.get(storage_config.default_workflow_type)  # type: ignore
        if self.is_eval and storage_config.default_eval_workflow_type:
            self.default_workflow_cls = WORKFLOWS.get(storage_config.default_eval_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(storage_config.default_reward_fn_type)  # type: ignore
        self.formatter = TaskFormatter(storage_config)
        self.latest_index = storage_config.index

    def write(self, data: List[Experience]) -> None:
        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            experience_models = [self.table_model_cls.from_experience(exp) for exp in data]
            session.add_all(experience_models)

    def read(self, batch_size: Optional[int] = None) -> List[Experience]:
        if self.stopped:
            raise StopIteration()

        exp_list = []
        batch_size = batch_size or self.batch_size  # type: ignore
        while len(exp_list) < batch_size:
            if len(exp_list):
                self.logger.info("waiting for experiences...")
                time.sleep(1)
            with retry_session(
                self.session, self.max_retry_times, self.max_retry_interval
            ) as session:
                # get a batch of experiences from the database
                experiences = (
                    session.query(self.table_model_cls)
                    .filter(self.table_model_cls.id > self.latest_index)
                    .order_by(asc(self.table_model_cls.id))
                    .limit(batch_size)
                    .all()
                )
                if experiences:
                    self.latest_index = experiences[-1].id
                exp_list.extend([self.table_model_cls.to_experience(exp) for exp in experiences])
        return exp_list


class SQLTaskStorage(SQLStorage):
    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        super().__init__(storage_config, config)
        self.is_eval = storage_config.is_eval
        self.default_workflow_cls = WORKFLOWS.get(storage_config.default_workflow_type)  # type: ignore
        if self.is_eval and storage_config.default_eval_workflow_type:
            self.default_workflow_cls = WORKFLOWS.get(storage_config.default_eval_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(storage_config.default_reward_fn_type)  # type: ignore
        self.formatter = TaskFormatter(storage_config)
        self.latest_index = storage_config.index

    def write(self, data: List[Dict]) -> None:
        with retry_session(
            self.session, self.max_retry_interval, self.max_retry_interval
        ) as session:
            tasks = [self.table_model_cls.from_dict(item) for item in data]
            session.add_all(tasks)

    def read(self, batch_size: Optional[int] = None) -> List[Task]:
        if self.stopped:
            raise StopIteration()
        batch_size = batch_size or self.batch_size
        with retry_session(
            self.session, self.max_retry_interval, self.max_retry_interval
        ) as session:
            query = (
                session.query(self.table_model_cls)
                .filter(self.table_model_cls.id > self.latest_index)
                .order_by(asc(self.table_model_cls.id))
                .limit(batch_size)
            )
            results = query.all()
            if not results:
                raise StopIteration()
            self.latest_index = results[-1].id
            return [self.formatter.format(item.raw_task) for item in results]
