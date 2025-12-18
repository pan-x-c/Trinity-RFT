from typing import List

from sqlalchemy.orm import sessionmaker

from trinity.buffer.schema import init_engine
from trinity.buffer.utils import retry_session
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


# TODO: implement an async version in the future
class HistoryRecorder:
    """Record chat history into the database."""

    def __init__(self, db_url: str, table_name: str):
        self.logger = get_logger()
        self.engine, self.table_model_cls = init_engine(
            db_url=db_url,
            table_name=table_name,
            schema_type="experience",
        )
        self.logger.info(f"Init SQL storage at {db_url}")
        self.session = sessionmaker(bind=self.engine)

    def record_history(self, experiences: List[Experience]) -> None:
        """Save experience to the database."""
        with retry_session(self.session) as db:
            exps = [self.table_model_cls.from_experience(exp) for exp in experiences]
            db.add_all(exps)

    def update_reward(
        self, reward: float, msg_ids: list, run_id: int, task_id: str
    ) -> List[Experience]:
        """Update reward for given response IDs and return the updated experiences.

        Args:
            reward (float): The reward value to be updated.
            msg_ids (list): List of message IDs to update.
            run_id (int): The run ID associated with the experiences.
            task_id (str): The task ID associated with the experiences.

        Returns:
            List[Experience]: List of updated experiences.

        Note:
            Only experiences that have not been consumed (consumed == 0) will be returned.
            For example, if you call this method multiple times with the same msg_ids, only
            the first call will return the updated experiences; subsequent calls will return
            an empty list.
        """
        with retry_session(self.session) as db:
            db.execute(
                self.table_model_cls.__table__.update()
                .where((self.table_model_cls.msg_id.in_(msg_ids)))
                .values(
                    reward=reward,
                    run_id=run_id,
                    task_id=task_id,
                    consumed=self.table_model_cls.consumed + 1,
                )
            )

        with retry_session(self.session) as db:
            results = db.execute(
                self.table_model_cls.__table__.select().where(
                    (
                        self.table_model_cls.msg_id.in_(msg_ids)
                        & (self.table_model_cls.consumed == 1)
                    )
                )
            ).all()
            updated_experiences = [self.table_model_cls.to_experience(row) for row in results]
        return updated_experiences
