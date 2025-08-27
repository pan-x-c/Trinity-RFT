"""SQLAlchemy models for different data."""

from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, Float, Integer, LargeBinary, Text
from sqlalchemy.orm import declarative_base

from trinity.common.experience import Experience
from trinity.utils.registry import Registry

SQL_SCHEMA = Registry("sql_schema")

Base = declarative_base()


@SQL_SCHEMA.register_module("task")
class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_task = Column(JSON, nullable=False)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(raw_task=dict)


@SQL_SCHEMA.register_module("experience")
class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    # for single turn
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    # for multi turn
    message_list = Column(JSON, nullable=True)
    reward = Column(Float, nullable=True)
    # serialized experience object
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            reward=experience.reward,
            prompt=experience.prompt_text,
            response=experience.response_text,
            message_list=experience.messages,
        )


@SQL_SCHEMA.register_module("sft")
class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            message_list=experience.messages,
        )


@SQL_SCHEMA.register_module("dpo")
class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    chosen_message_list = Column(JSON, nullable=True)
    rejected_message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            chosen_message_list=experience.chosen_messages,
            rejected_message_list=experience.rejected_messages,
        )


def create_dynamic_table(table_name: str, sql_schema_type: Optional[str]) -> Any:
    """Create a dynamic table based on the provided algorithm type and table name."""
    if sql_schema_type is None:
        sql_schema_type = "task"

    print(f"table_name: {table_name}, sql_schema_type: {sql_schema_type}")

    base_class = SQL_SCHEMA.get(sql_schema_type)

    table_attrs = {
        "__tablename__": table_name,
        "__abstract__": False,
        "__table_args__": {"extend_existing": True},
    }
    print(base_class)
    return type(table_name, (base_class,), table_attrs)
