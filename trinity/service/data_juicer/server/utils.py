from typing import Any, Dict, List, Optional

from data_juicer.config import get_init_configs, prepare_side_configs
from jsonargparse import Namespace
from pydantic import BaseModel, model_validator


class OperatorModel(BaseModel):
    """Model for individual operator configuration.

    Example:

    .. code-block:: python

        {
            "cleaner": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    """

    __root__: Dict[str, Dict[str, Any]]


class DataJuicerConfigModel(BaseModel):
    operators: Optional[List[OperatorModel]]
    config_path: Optional[str]

    @model_validator
    def check_priority(cls, values):
        if not (values.get("config_path") or values.get("operators")):
            raise ValueError("Must provide at least one of config_path or operators.")
        return values


def parse_config(config: DataJuicerConfigModel) -> Namespace:
    """Convert Trinity config to DJ config"""
    if config.operators is not None:
        dj_config = Namespace(process=[op.model_dump() for op in config.operators])
        dj_config = get_init_configs(dj_config)
    elif config.config_path is not None:
        dj_config = prepare_side_configs(config.config_path)
        dj_config = get_init_configs(dj_config)
    else:
        raise ValueError("At least one of operators, config_path, description should be provided.")

    return dj_config
