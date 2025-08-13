from typing import Any, Dict, List, Optional

from data_juicer.config import get_init_configs, prepare_side_configs
from jsonargparse import Namespace
from pydantic import BaseModel, model_validator


class DataJuicerConfigModel(BaseModel):
    operators: Optional[List[Dict[str, Dict[str, Any]]]] = None
    config_path: Optional[str] = None

    @model_validator(mode="after")
    def check_priority(self):
        if not (self.config_path or self.operators):
            raise ValueError("Must provide at least one of config_path or operators.")
        return self


def parse_config(config: DataJuicerConfigModel) -> Namespace:
    """Convert Trinity config to DJ config"""
    if config.operators is not None:
        dj_config = Namespace(process=[op for op in config.operators])
        dj_config = get_init_configs(dj_config)
    elif config.config_path is not None:
        dj_config = prepare_side_configs(config.config_path)
        dj_config = get_init_configs(dj_config)
    else:
        raise ValueError("At least one of operators, config_path, description should be provided.")

    return dj_config
