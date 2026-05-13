"""Allocator module for managing inference engines."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
from copy import deepcopy

import ray
from ray.util.placement_group import placement_group, placement_group_table, PlacementGroup


from trinity.common.config import ExplorerConfig, InferenceModelConfig
from trinity.common.models.model import ModelWrapper
from trinity.utils.log import get_logger


@dataclass
class BundleResult:
    """Data class for storing the result of bundle allocation."""

    bundles: List[Dict[str, float]]
    actor_bundle_map: Dict[str, int]
    bundle_actor_map: Dict[int, str]


class Allocator:
    """Allocator class for managing inference engines."""

    def __init__(self, config: ExplorerConfig):
        """Initialize the Allocator."""
        self.config = config
        self.logger = get_logger(__name__)

    def get_actor_name(self, role: str, engine_id: int, node_id: int) -> str:
        """Generate a unique actor name based on the model config, engine ID, and node ID."""
        return f"{self.config.name}_{role}_model_{engine_id}_{node_id}"

    def allocate_bundles(self) -> BundleResult:
        """Allocate bundles for the rollout model and auxiliary models based on the configuration."""
        rollout_model = self.config.rollout_model
        auxiliary_models = self.config.auxiliary_models
        model_configs = [("rollout", rollout_model)] + [
            (f"auxiliary_{index}", model) for index, model in enumerate(auxiliary_models)
        ]
        bundles: List[Dict[str, float]] = []
        engine_bundle_map: Dict[str, int] = {}
        bundle_engine_map: Dict[int, str] = {}
        bundle_id = 0
        for role, config in model_configs:
            gpus_per_bundle = config.tensor_parallel_size // config.nnodes
            for engine_id in range(config.engine_num):
                for node_id in range(config.nnodes):
                    bundles.append({"GPU": float(gpus_per_bundle), "CPU": 1})
                    actor_name = self.get_actor_name(role, engine_id, node_id)
                    engine_bundle_map[actor_name] = bundle_id
                    bundle_engine_map[bundle_id] = actor_name
                    bundle_id += 1
        return BundleResult(
            bundles=bundles, actor_bundle_map=engine_bundle_map, bundle_actor_map=bundle_engine_map
        )

    def analysis_placement_group(self, pg: PlacementGroup, bundle_result: BundleResult):
        bundle_node_map = placement_group_table(pg)["bundles_to_node_id"]
        node_bundle_map = defaultdict(list)
        for bundle_id, node_id in bundle_node_map.items():
            node_bundle_map[node_id].append(bundle_id)
        for node_id, bundle_ids in node_bundle_map.items():
            self.logger.info("Node %s bundles:", node_id)
            for bundle_id in bundle_ids:
                actor_name = bundle_result.bundle_actor_map[bundle_id]
                self.logger.info("  > Bundle %s: Actor %s", bundle_id, actor_name)

    async def create_engine(
        self, config: InferenceModelConfig, role: str, engine_id: int
    ) -> ModelWrapper:
        config = deepcopy(config)
        config.engine_id = engine_id
        actor_bundle_lists = []
        for node_id in range(config.nnodes):
            actor_name = self.get_actor_name(role, engine_id, node_id)
            actor_bundle_lists.append((actor_name, self.bundle_result.actor_bundle_map[actor_name]))
        if config.engine_type.startswith("vllm"):
            from trinity.common.models.vllm_model import vLLMRolloutModel

            return await vLLMRolloutModel.get_wrapper(config, self.pg, actor_bundle_lists)
        elif config.engine_type == "sglang":
            from trinity.common.models.sglang_model import SGLangRolloutModel

            pass
        elif config.engine_type == "tinker":
            pass
        else:
            pass

    async def create_all_models(self) -> Tuple[List[ModelWrapper], List[List[ModelWrapper]]]:
        """Create all model actors for the rollout model and auxiliary models based on the configuration."""
        self.bundle_result = self.allocate_bundles()
        self.pg = placement_group(self.bundle_result.bundles, strategy="PACK")
        await self.pg.ready()
        self.analysis_placement_group(self.pg, self.bundle_result)
        # create rollout_models
        rollout_models = []
        tasks = []
        for engine_id in range(self.config.rollout_model.engine_num):
            tasks.append(
                asyncio.create_task(
                    self.create_engine(self.config.rollout_model, "rollout", engine_id)
                )
            )
        # create auxiliary models
        for index, auxiliary_model_config in enumerate(self.config.auxiliary_models):
            for engine_id in range(auxiliary_model_config.engine_num):
                tasks.append(
                    asyncio.create_task(
                        self.create_engine(auxiliary_model_config, f"auxiliary_{index}", engine_id)
                    )
                )
        # wait for all models to be created
        results = await asyncio.gather(*tasks)
        rollout_models: List[ModelWrapper] = results[: self.config.rollout_model.engine_num]
        auxiliary_models: List[List[ModelWrapper]] = [
            results[
                self.config.rollout_model.engine_num
                + sum(
                    self.config.auxiliary_models[i].engine_num for i in range(index)
                ) : self.config.rollout_model.engine_num
                + sum(self.config.auxiliary_models[i].engine_num for i in range(index + 1))
            ]
            for index in range(len(self.config.auxiliary_models))
        ]
        return rollout_models, auxiliary_models

    def get_model(
        self, config: InferenceModelConfig, role: str, engine_id: int
    ) -> ModelWrapper:
        """Get the model actor for the given role and engine ID."""
        actor_name = self.get_actor_name(role, engine_id, 0)
        try:
            return ray.get_actor(actor_name, namespace=config.ray_namespace)
        except ValueError:
            self.logger.error("Actor %s not found. Make sure the model is created.", actor_name)
            raise
