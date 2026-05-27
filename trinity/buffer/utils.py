import asyncio
import time
import traceback
from contextlib import contextmanager
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from trinity.utils.log import get_logger


@contextmanager
def retry_session(session_maker, max_retry_times: int = 2, max_retry_interval: float = 1.0):
    """A context manager for a single session lifecycle."""
    del max_retry_times, max_retry_interval
    session = session_maker()
    try:
        yield session
        session.commit()
    except StopIteration:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_with_retry_session(
    session_maker,
    operation: Callable[[Session], Any],
    max_retry_times: int = 2,
    max_retry_interval: float = 1.0,
) -> Any:
    """Run a database operation with session retry around a single transaction."""
    logger = get_logger(__name__)
    max_retry_times = max(1, max_retry_times)

    for attempt in range(max_retry_times):
        try:
            with retry_session(session_maker) as session:
                return operation(session)
        except StopIteration:
            raise
        except Exception as exc:
            trace_str = traceback.format_exc()
            logger.warning(
                "Attempt %s failed, retrying in %s seconds...",
                attempt + 1,
                max_retry_interval,
            )
            logger.warning("trace = %s", trace_str)
            if attempt < max_retry_times - 1:
                time.sleep(max_retry_interval)
                continue

            logger.error("Max retry attempts reached, raising exception.")
            raise exc

    raise RuntimeError("run_with_retry_session exhausted without raising")


def to_async_url(url: str) -> str:
    """Convert a synchronous DB URL to its async dialect equivalent."""
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("mysql://"):
        return url.replace("mysql://", "mysql+aiomysql://", 1)
    return url


async def async_run_with_retry_session(
    session_maker: async_sessionmaker,
    operation: Callable[[AsyncSession], Awaitable[Any]],
    max_retry_times: int = 2,
    max_retry_interval: float = 1.0,
) -> Any:
    """Run an async database operation with session retry."""
    logger = get_logger(__name__)
    max_retry_times = max(1, max_retry_times)

    for attempt in range(max_retry_times):
        async with session_maker() as session:
            try:
                async with session.begin():
                    result = await operation(session)
                return result
            except StopAsyncIteration:
                raise
            except Exception as exc:
                logger.warning(
                    "Async attempt %s failed, retrying in %s seconds...",
                    attempt + 1,
                    max_retry_interval,
                )
                logger.warning("trace = %s", traceback.format_exc())
                if attempt < max_retry_times - 1:
                    await asyncio.sleep(max_retry_interval)
                    continue
                logger.error("Max retry attempts reached, raising exception.")
                raise exc

    raise RuntimeError("async_run_with_retry_session exhausted without raising")
