"""Schema for SQLAlchemy models."""

from typing import Any, Optional

from sqlalchemy import JSON, Column, Float, Integer, LargeBinary, String, Text
from sqlalchemy.ext.declarative import declarative_base

from trinity.common.experience import Experience

Base = declarative_base()


class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_task = Column(JSON, nullable=False)
    workflow_type = Column(String, nullable=True)
    reward_type = Column(String, nullable=True)


class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

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
            serialized_exp=experience.serialize(),
            reward=experience.reward,
            prompt=experience.prompt_text,
            response=experience.response_text,
            message_list=experience.messages,
        )


class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

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
            serialized_exp=experience.serialize(),
            message_list=experience.messages,
        )


class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    chosen_message_list = Column(JSON, nullable=True)
    rejected_message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.serialized_exp)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            serialized_exp=experience.serialize(),
            chosen_message_list=experience.chosen_messages,
            rejected_message_list=experience.rejected_messages,
        )


def create_dynamic_table(algorithm_type: Optional[str], table_name: str) -> Any:
    """Create a dynamic table based on the provided algorithm type and table name."""
    if algorithm_type is None:
        base_class = TaskModel
    else:
        from trinity.algorithm.algorithm import ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(algorithm_type)
        base_class = algorithm.schema

    table_attrs = {
        "__tablename__": table_name,
    }

    return type(table_name, (base_class,), table_attrs)
