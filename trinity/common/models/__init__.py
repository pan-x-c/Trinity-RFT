import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import ray
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from trinity.common.config import ExplorerConfig, InferenceModelConfig
from trinity.utils.log import get_logger


@dataclass
class InferenceEngineAllocationResult:
    bundles: List[Dict[str, float]]
    engine_bundle_map: Dict[str, int]


def allocate_bundles(explorer_config: ExplorerConfig) -> InferenceEngineAllocationResult:
    rollout_model = explorer_config.rollout_model
    auxiliary_models = explorer_config.auxiliary_models
    explorer_name = explorer_config.name
    model_configs = [
        (f"{explorer_name}_rollout_model_{rollout_model.name or 0}", rollout_model)
    ] + [
        (f"{explorer_name}_auxiliary_model_{model.name or index}", model)
        for index, model in enumerate(auxiliary_models)
    ]
    bundles: List[Dict[str, float]] = []
    engine_bundle_map: Dict[str, int] = {}
    bundle_id = 0
    for label, config in model_configs:
        gpus_per_bundle = config.tensor_parallel_size // config.nnodes
        for engine_id in range(config.engine_num):
            for node_id in range(config.nnodes):
                bundles.append({"GPU": float(gpus_per_bundle), "CPU": 1})
                engine_bundle_map[f"{label}_{engine_id}_{node_id}"] = bundle_id
                bundle_id += 1
    return InferenceEngineAllocationResult(bundles=bundles, engine_bundle_map=engine_bundle_map)


class Allocator:
    """Allocate placement-group bundles for inference engines."""

    def __init__(
        self,
        explorer_config: ExplorerConfig,
    ) -> None:
        rollout_model = explorer_config.rollout_model
        auxiliary_models = explorer_config.auxiliary_models
        explorer_name = explorer_config.name
        self.logger = get_logger(__name__, in_ray_actor=True)
        self._engine_bundles: Dict[str, List[List[int]]] = {}
        self.engine_bundle_map: Dict[str, List[int]] = {}
        bundles: List[Dict[str, float]] = []
        self._configs = [
            (f"{explorer_name}_rollout_model_{rollout_model.name or 0}", rollout_model)
        ] + [
            (f"{explorer_name}_auxiliary_model_{model.name or index}", model)
            for index, model in enumerate(auxiliary_models)
        ]
        bundle_id = 0
        for label, config in self._configs:
            self._validate_config(config=config, label=label)
            gpus_per_bundle = config.tensor_parallel_size // config.nnodes
            for engine_id in range(config.engine_num):
                for node_id in range(config.nnodes):
                    bundles.append({"GPU": float(gpus_per_bundle)})
                    self.logger.info(
                        "Prepared bundle %d for %s engine %d with %d GPUs.",
                        bundle_id,
                        label,
                        engine_id,
                        gpus_per_bundle,
                    )
                    bundle_id += 1
                self.engine_bundle_map[f"{label}_{engine_id}"] = list(
                    range(bundle_id - config.nnodes, bundle_id)
                )

        self.pg = placement_group(bundles, strategy="PACK")
        ray.get(self.pg.ready())

        bundle_node_map = placement_group_table(self.pg)["bundles_to_node_id"]
        node_bundle_map = defaultdict(list)
        for bundle_id, node_id in bundle_node_map.items():
            node_bundle_map[node_id].append(bundle_id)

        self.node_bundle_list = [sorted(value) for value in node_bundle_map.values()]
        self.node_list = [key for key in node_bundle_map.keys()]
        self._bind_engine_bundles()

    @staticmethod
    def _validate_config(config: InferenceModelConfig, label: str) -> None:
        if config.engine_num < 1:
            raise ValueError(f"`{label}.engine_num` must be >= 1, but got {config.engine_num}.")
        if config.tensor_parallel_size < 1:
            raise ValueError(
                f"`{label}.tensor_parallel_size` must be >= 1, but got "
                f"{config.tensor_parallel_size}."
            )
        if config.nnodes < 1:
            raise ValueError(f"`{label}.nnodes` must be >= 1, but got {config.nnodes}.")
        if config.tensor_parallel_size < config.nnodes:
            raise ValueError(
                f"`{label}.tensor_parallel_size` ({config.tensor_parallel_size}) must be >= "
                f"`{label}.nnodes` ({config.nnodes})."
            )
        if config.tensor_parallel_size % config.nnodes != 0:
            raise ValueError(
                f"`{label}.tensor_parallel_size` ({config.tensor_parallel_size}) must be "
                f"divisible by `nnodes` ({config.nnodes})."
            )

    def _bind_engine_bundles(self) -> None:
        for label, config in self._configs:
            engine_bundles = []
            for _ in range(config.engine_num):
                candidate_nodes = []
                for node_index in range(len(self.node_bundle_list)):
                    if self.node_bundle_list[node_index]:
                        candidate_nodes.append((node_index, self.node_bundle_list[node_index][0]))
                    if len(candidate_nodes) == config.nnodes:
                        break

                if len(candidate_nodes) != config.nnodes:
                    raise ValueError(
                        "Bundle allocation error, unable to allocate an engine for "
                        f"`{label}` with tensor_parallel_size={config.tensor_parallel_size}, "
                        f"nnodes={config.nnodes}."
                    )

                bundle_list = []
                allocation_log = []
                for node_index, bundle_id in candidate_nodes:
                    self.node_bundle_list[node_index].remove(bundle_id)
                    bundle_list.append(bundle_id)
                    allocation_log.append((self.node_list[node_index], [bundle_id]))

                engine_bundles.append(bundle_list)
                self.logger.info("Allocate bundles for %s: %s.", label, allocation_log)

            self._engine_bundles[label] = engine_bundles

    def allocate(self, label: str, engine_id: int) -> List[int]:
        if label not in self._engine_bundles:
            raise ValueError(f"Unknown allocation label: {label}.")
        if engine_id < 0 or engine_id >= len(self._engine_bundles[label]):
            raise ValueError(f"`engine_id` out of range for `{label}`: {engine_id}.")

        config = dict(self._configs)[label]
        gpus_per_bundle = config.tensor_parallel_size // config.nnodes
        bundle_list = self._engine_bundles[label][engine_id]
        if len(bundle_list) != config.nnodes:
            raise ValueError(
                f"Allocated {len(bundle_list)} bundles for `{label}` engine {engine_id}, "
                f"expected {config.nnodes}."
            )
        if gpus_per_bundle * len(bundle_list) != config.tensor_parallel_size:
            raise ValueError(
                f"Allocated bundles {bundle_list} for `{label}` engine {engine_id} do not "
                f"match tensor_parallel_size={config.tensor_parallel_size}."
            )
        return bundle_list


# def create_explorer_models(
#     config: Config,
# ) -> Tuple[List, List[List]]:
#     """Create rollout_models and auxiliary_models.

#     Args:
#         config: The trinity configuration.
#     Returns:
#         Tuple[List, List[List]]: The rollout_models and auxiliary_models.
#     """
#     rollout_engines = []
#     if config.explorer.rollout_model.engine_type.startswith("vllm"):
#         from trinity.common.models.vllm_model import vLLMRolloutModel

#         engine_cls = vLLMRolloutModel
#     elif config.explorer.rollout_model.engine_type == "sglang":
#         from trinity.common.models.sglang_model import SGLangRolloutModel

#         engine_cls = SGLangRolloutModel
#     elif config.explorer.rollout_model.engine_type == "external":
#         rollout_engines = create_external_models(
#             config=config.explorer.rollout_model,
#             actor_name=f"{config.explorer.name}_rollout_model",
#         )
#         auxiliary_engines = []
#         for i, model_config in enumerate(config.explorer.auxiliary_models):
#             engines = create_external_models(
#                 config=model_config,
#                 actor_name=f"{config.explorer.name}_auxiliary_model_{model_config.name or i}",
#             )
#             auxiliary_engines.append(engines)
#         return rollout_engines, auxiliary_engines
#     elif config.explorer.rollout_model.engine_type == "tinker":
#         from trinity.common.models.tinker_model import TinkerModel

#         engine_cls = TinkerModel
#         namespace = config.ray_namespace
#         rollout_engines = [
#             ray.remote(engine_cls)
#             .options(
#                 name=f"{config.explorer.name}_rollout_model_{i}",
#                 namespace=namespace,
#             )
#             .remote(
#                 config=config.explorer.rollout_model,
#             )
#             for i in range(config.explorer.rollout_model.engine_num)
#         ]
#         auxiliary_engines = [
#             [
#                 ray.remote(engine_cls)
#                 .options(
#                     name=f"{config.explorer.name}_auxiliary_model_{model_config.name or i}_{j}",
#                     namespace=namespace,
#                 )
#                 .remote(
#                     config=config.explorer.auxiliary_models[i],
#                 )
#                 for j in range(model_config.engine_num)
#             ]
#             for i, model_config in enumerate(config.explorer.auxiliary_models)
#         ]
#         return rollout_engines, auxiliary_engines
#     else:
#         raise ValueError(f"Unknown engine type: {config.explorer.rollout_model.engine_type}")


def create_inference_models(
    config: InferenceModelConfig,
    allocator: Allocator,
    allocation_label: str,
    actor_name: str,
) -> List:
    model_cls = None
    if config.engine_type == "sglang":
        from trinity.common.models.sglang_model import SGLangRolloutModel

        model_cls = SGLangRolloutModel
    else:
        from trinity.common.models.vllm_model import vLLMRolloutModel

        model_cls = vLLMRolloutModel

    models = []
    for i in range(config.engine_num):
        bundles_for_engine = allocator.allocate(allocation_label, i)
        model_config = deepcopy(config)
        model_config.engine_id = i
        models.append(
            ray.remote(model_cls)
            .options(
                name=f"{actor_name}_{i}",
                num_cpus=0,
                num_gpus=model_config.tensor_parallel_size,
                namespace=model_config.ray_namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=allocator.pg,
                    placement_group_capture_child_tasks=True,
                    # TODO: allocate a specific bundle
                    placement_group_bundle_index=bundles_for_engine[0],
                ),
            )
            .remote(
                config=model_config,
            )
        )
    return models


def create_sglang_inference_models(
    config: InferenceModelConfig,
    allocator: Allocator,
    allocation_label: str,
    actor_name: str,
) -> List:
    from trinity.common.models.sglang_model import SGLangRolloutModel

    models = []
    for i in range(config.engine_num):
        bundles_for_engine = allocator.allocate(allocation_label, i)
        model_config = deepcopy(config)
        model_config.engine_id = i
        models.append(
            ray.remote(SGLangRolloutModel)
            .options(
                name=f"{actor_name}_{i}",
                num_cpus=0,
                num_gpus=0 if model_config.tensor_parallel_size > 1 else 1,
                namespace=model_config.ray_namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=allocator.pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundles_for_engine[0],
                ),
            )
            .remote(
                config=model_config,
            )
        )
    return models


def create_external_models(
    config: InferenceModelConfig,
    actor_name: str,
) -> List:
    from trinity.common.models.external_model import ExternalModel

    # Ensure external-model env vars are propagated to Ray workers.
    env_vars = {}
    base_url_env = config.external_model_config.base_url_env
    api_key_env = config.external_model_config.api_key_env
    if base_url_env and base_url_env in os.environ:
        env_vars[base_url_env] = os.environ[base_url_env]
    if api_key_env and api_key_env in os.environ:
        env_vars[api_key_env] = os.environ[api_key_env]

    models = []
    for i in range(config.engine_num):
        model_config = deepcopy(config)
        model_config.engine_id = i
        models.append(
            ray.remote(ExternalModel)
            .options(
                name=f"{actor_name}_{i}",
                num_cpus=0,
                num_gpus=0,
                namespace=model_config.ray_namespace,
                runtime_env={"env_vars": env_vars},
            )
            .remote(
                config=model_config,
            )
        )
    return models
