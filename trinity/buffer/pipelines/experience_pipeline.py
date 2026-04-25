import asyncio
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional

from trinity.buffer.buffer import BufferWriter, get_buffer_reader, get_buffer_writer
from trinity.buffer.operators.experience_operator import (
    create_operators,
    ensure_v1_operator,
)
from trinity.buffer.storage.queue import is_database_url, is_json_file
from trinity.common.config import (
    AlgorithmConfig,
    Config,
    ExperiencePipelineConfig,
    StorageConfig,
)
from trinity.common.constants import SELECTOR_METRIC, StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


def get_input_buffers(pipeline_config: ExperiencePipelineConfig) -> Dict:
    """Get input buffers for the experience pipeline."""
    input_buffers = {}
    for input_name, input_config in pipeline_config.inputs.items():
        buffer_reader = get_buffer_reader(input_config)
        input_buffers[input_name] = buffer_reader
    return input_buffers


class ExperiencePipeline:
    """
    A class to process experiences.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(f"{config.explorer.name}_experience_pipeline", in_ray_actor=True)
        load_plugins()
        self.config = config
        self.input_store = self._init_input_storage(config.data_processor.experience_pipeline)  # type: ignore [arg-type]
        self.output = get_buffer_writer(
            config.buffer.trainer_input.experience_buffer,  # type: ignore [arg-type]
        )
        self.auxiliary_model_wrappers = {}
        self.auxiliary_models = {}
        self.staged_task_payloads = defaultdict(dict)

    def _init_input_storage(
        self,
        pipeline_config: ExperiencePipelineConfig,
    ) -> Optional[BufferWriter]:
        """Initialize the input storage if it is not already set."""
        if pipeline_config.save_input:
            if pipeline_config.input_save_path is None:
                raise ValueError("input_save_path must be set when save_input is True.")
            elif is_json_file(pipeline_config.input_save_path):
                return get_buffer_writer(
                    StorageConfig(
                        storage_type=StorageType.FILE.value,
                        path=pipeline_config.input_save_path,
                        schema_type="experience",
                        wrap_in_ray=False,
                    ),
                )
            elif is_database_url(pipeline_config.input_save_path):
                return get_buffer_writer(
                    StorageConfig(
                        name="pipeline_input",
                        storage_type=StorageType.SQL.value,
                        path=pipeline_config.input_save_path,
                        schema_type="experience",
                        wrap_in_ray=False,
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported save_input format: {pipeline_config.save_input}. "
                    "Only JSON file path or SQLite URL is supported."
                )
        else:
            return None

    def _set_algorithm_operators(self, algorithm_config: AlgorithmConfig) -> None:
        """Add algorithm-specific operators to the pipeline."""
        from trinity.algorithm import ADVANTAGE_FN, ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(algorithm_config.algorithm_type)
        if not algorithm.compute_advantage_in_trainer and algorithm_config.advantage_fn:
            advantage_fn_cls = ADVANTAGE_FN.get(algorithm_config.advantage_fn)
            assert (
                advantage_fn_cls is not None
            ), f"AdvantageFn {algorithm_config.advantage_fn} not found."
            assert (
                not advantage_fn_cls.compute_in_trainer()
            ), f"AdvantageFn {algorithm_config.advantage_fn} can only be computed in the trainer, please check your implementation."
            operator = ensure_v1_operator(advantage_fn_cls(**algorithm_config.advantage_fn_args))
            operator.set_auxiliary_model(self.auxiliary_models)
            self.operators.append(operator)

    async def prepare(self) -> None:
        from trinity.common.models import get_auxiliary_model_wrappers

        # make sure auxiliary models are ready before creating operators
        model_wrappers = await get_auxiliary_model_wrappers(self.config)
        self.auxiliary_model_wrappers.update(model_wrappers)
        self.auxiliary_models = (
            {
                model_name: [
                    model_wrapper.get_openai_async_client() for model_wrapper in model_wrappers
                ]
                for model_name, model_wrappers in self.auxiliary_model_wrappers.items()
            }
            if self.auxiliary_model_wrappers
            else {}
        )
        await self.output.acquire()
        try:
            self.operators = create_operators(
                self.config.data_processor.experience_pipeline.operators,
                self.auxiliary_models,
            )
            self._set_algorithm_operators(self.config.algorithm)
            for operator in self.operators:
                await operator.prepare()
        except Exception as e:
            self.logger.error(f"Failed to create experience operators: {traceback.format_exc()}")
            raise e

    async def process(self, exp_bytes: bytes) -> Dict:
        """Process a batch of experiences.

        Args:
            exp_bytes (bytes): Serialized experiences to process. These experiences are typically generated by an explorer in one step.

        Returns:
            Dict: A dictionary containing metrics collected during the processing of experiences.
        """
        exps = Experience.deserialize_many(exp_bytes)
        return await self._process_experiences(exps)

    async def process_serialized_chunks(self, exp_chunks: list[bytes]) -> Dict:
        """Process a batch assembled from multiple serialized task payloads."""
        exps = []
        for exp_bytes in exp_chunks:
            if not exp_bytes:
                continue
            exps.extend(Experience.deserialize_many(exp_bytes))
        return await self._process_experiences(exps)

    async def stage_task_payloads(
        self, batch_id, task_id: int, exp_chunks: list[bytes]
    ) -> Optional[str]:
        """Stage serialized payload chunks for one completed task."""
        valid_chunks = [chunk for chunk in exp_chunks if chunk]
        if not valid_chunks:
            return None
        self.staged_task_payloads[batch_id][task_id] = valid_chunks
        return f"{batch_id}:{task_id}"

    async def finalize_batch(self, batch_id, task_ids: Optional[list[int]] = None) -> Dict:
        """Finalize a staged batch and process all staged task payloads."""
        batch_payloads = self.staged_task_payloads.get(batch_id, {})
        if not batch_payloads:
            return await self._process_experiences([])

        selected_task_ids = task_ids or list(batch_payloads.keys())
        exp_chunks = []
        for task_id in selected_task_ids:
            exp_chunks.extend(batch_payloads.pop(task_id, []))

        if batch_id in self.staged_task_payloads and not self.staged_task_payloads[batch_id]:
            del self.staged_task_payloads[batch_id]

        return await self.process_serialized_chunks(exp_chunks)

    async def take_staged_task_payloads(self, batch_id, task_ids: list[int]) -> list[bytes]:
        """Drain staged payload chunks for selected tasks without processing them."""
        batch_payloads = self.staged_task_payloads.get(batch_id, {})
        exp_chunks = []
        for task_id in task_ids:
            exp_chunks.extend(batch_payloads.pop(task_id, []))

        if batch_id in self.staged_task_payloads and not self.staged_task_payloads[batch_id]:
            del self.staged_task_payloads[batch_id]

        return exp_chunks

    async def abort_batch(self, batch_id) -> None:
        """Discard any staged payloads for a batch."""
        self.staged_task_payloads.pop(batch_id, None)

    async def _process_experiences(self, exps: list[Experience]) -> Dict:
        st = time.time()
        if self.input_store is not None:
            await self.input_store.write_async(exps)

        if not hasattr(self, "operators"):
            raise RuntimeError(
                "ExperiencePipeline is not prepared. Please call prepare() before processing experiences."
            )

        metrics = {}

        # Process experiences through operators
        for idx, operator in enumerate(self.operators):
            with Timer(
                metrics, f"time/experience_pipeline/operator/{idx}_{operator.__class__.__name__}"
            ):
                exps, metric = await operator.process(exps)
                metrics.update(metric)
        metrics["experience_count"] = len(exps)

        # Write processed experiences to output buffer
        with Timer(metrics, "time/experience_pipeline/write"):
            await self.output.write_async(exps)
        metrics["time/experience_pipeline/total"] = time.time() - st

        # prefix metrics keys with 'pipeline/'
        result_metrics = {}
        for key, value in metrics.items():
            if key.startswith("time/"):
                result_metrics[key] = value
            elif isinstance(value, (int, float)):
                result_metrics[f"experience_pipeline/{key}"] = float(value)
        if SELECTOR_METRIC in metrics:
            result_metrics[SELECTOR_METRIC] = metrics[SELECTOR_METRIC]

        return result_metrics

    async def close(self) -> None:
        try:
            if self.output:
                await self.output.release()
            if hasattr(self, "operators") and self.operators:
                await asyncio.gather(*[operator.close() for operator in self.operators])
        except Exception as e:
            self.logger.error(f"Failed to release output buffer: {e}")
