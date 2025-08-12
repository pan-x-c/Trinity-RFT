from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class OperatorModel(BaseModel):
    __root__: Dict[str, Dict[str, Any]]


class DataJuicerConfigModel(BaseModel):
    config_path: Optional[str]
    operators: Optional[List[OperatorModel]]
    description: Optional[str]

    @model_validator
    def check_priority(cls, values):
        if not (values.get("config_path") or values.get("operators") or values.get("description")):
            raise ValueError("Must provide at least one of config_path, operators, or description.")
        return values
