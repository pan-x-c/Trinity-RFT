"""A Ray compatible logging module with actor-scope logger support."""
import contextvars
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from trinity.common.constants import LOG_DIR_ENV_VAR, LOG_LEVEL_ENV_VAR

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_ray_logger_ctx = contextvars.ContextVar("ray_logger", default=None)


def get_ray_actor_logger(name: str = None, level: Optional[int] = None) -> logging.Logger:
    return get_logger(name, level, in_ray_actor=True)


def get_logger(
    name: str = None, level: Optional[int] = None, in_ray_actor: bool = False
) -> logging.Logger:
    """
    Get a logger instance for current actor.

    Args:
        name (str): The name of the logger
        level (int): The logging level
        in_ray_actor (bool): Whether the logger is used within a Ray actor

    """
    level = level or getattr(logging, os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper())

    if in_ray_actor:
        logger = _ray_logger_ctx.get()
        if logger is not None:
            return logger

    logger = logging.getLogger(f"trinity.{name}")

    logger.setLevel(level)
    logger.handlers.clear()

    # stream log
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if in_ray_actor:
        # file log
        log_dir = os.environ.get(LOG_DIR_ENV_VAR, None)
        if log_dir is None:
            # File logging is disabled if LOG_DIR_ENV_VAR not provided
            return logger
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(file_path, encoding="utf-8", maxBytes=64 * 1024 * 1024)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.propagate = False
        _ray_logger_ctx.set(logger)  # type: ignore[arg-type]
    return logger
