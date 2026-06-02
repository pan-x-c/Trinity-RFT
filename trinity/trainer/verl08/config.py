from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trinity.common.config import Config


@dataclass
class VERLConfig:
    actor_rollout_ref: ActorRolloutRefConfig
    trainer: TrainerConfig
    algorithm: Dict = field(default_factory=dict)
    global_profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


@dataclass
class ActorRolloutRefConfig:
    model: ModelConfig
    actor: ActorConfig
    critic: CriticConfig


@dataclass
class TrainerConfig:
    gpu_per_node: int
    node_num: int


@dataclass
class ActorConfig:
    use_kl_loss: bool = False


@dataclass
class ProfilerConfig:
    steps: Optional[int] = None
    tools: str = "nsys"


@dataclass
class CriticConfig:
    enable: bool = False


@dataclass
class ModelConfig:
    path: str
    use_shm: bool = False
    # lora config
    lora_rank: int = 0
    lora_adapter_path: str = ""


def build_verl_config(global_config: Config) -> VERLConfig:
    """Extract and build the veRL-specific configuration from the global config."""
    raise NotImplementedError(
        "This function is a placeholder. Implement config extraction logic here."
    )
