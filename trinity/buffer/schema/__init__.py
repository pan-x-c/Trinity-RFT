from trinity.buffer.schema.sql_schema import init_engine
from trinity.utils.registry import Registry

FORMATTER: Registry = Registry(
    "formatter",
    {
        "task": "trinity.buffer.schema.formatter.TaskFormatter",
        "sft": "trinity.buffer.schema.formatter.SFTFormatter",
        "dpo": "trinity.buffer.schema.formatter.DPOFormatter",
    },
)

SQL_SCHEMA: Registry = Registry(
    "sql_schema",
    {
        "task": "trinity.buffer.schema.sql_schema.TaskModel",
    },
)

__all__ = ["init_engine", "FORMATTER", "SQL_SCHEMA"]
