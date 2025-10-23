import os
import time
from contextlib import contextmanager

from trinity.common.constants import CHECKPOINT_JOB_DIR_ENV_VAR, StorageType
from trinity.utils.log import get_logger


@contextmanager
def retry_session(session_maker, max_retry_times: int, max_retry_interval: float):
    """A Context manager for retrying session."""
    logger = get_logger(__name__)
    for attempt in range(max_retry_times):
        try:
            session = session_maker()
            yield session
            session.commit()
            break
        except StopIteration as e:
            raise e
        except Exception as e:
            import traceback

            trace_str = traceback.format_exc()
            session.rollback()
            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {max_retry_interval} seconds..."
            )
            logger.warning(f"trace = {trace_str}")
            if attempt < max_retry_times - 1:
                time.sleep(max_retry_interval)
            else:
                logger.error("Max retry attempts reached, raising exception.")
                raise e
        finally:
            session.close()


def default_storage_path(storage_name: str, storage_type: StorageType) -> str:
    checkpoint_dir = os.environ.get(CHECKPOINT_JOB_DIR_ENV_VAR, None)
    if checkpoint_dir is None:
        raise ValueError(
            f"Environment variable {CHECKPOINT_JOB_DIR_ENV_VAR} is not set. "
            "This should not happen when using `trinity run` command."
        )
    storage_dir = os.path.join(checkpoint_dir, "buffer")
    os.makedirs(storage_dir, exist_ok=True)
    if storage_type == StorageType.SQL:
        return "sqlite:///" + os.path.join(storage_dir, f"{storage_name}.db")
    else:
        return os.path.join(storage_dir, f"{storage_name}.jsonl")
