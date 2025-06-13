"""Filed based buffer reader."""

from itertools import islice
from typing import List, Optional

import datasets
import transformers
from datasets import Dataset, load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import AlgorithmType, PromptType, ReadStrategy, TaskType
from trinity.common.experience import Experience
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.registry import Registry

FILE_READERS = Registry("file_readers")


@FILE_READERS.register_module(AlgorithmType.SFT.value)
class _HFBatchReader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.current_batch_size = None

    def set_offset(self, offset: int) -> None:
        self.iter = self.dataset.iter(offset)

    def read_batch(self, batch_size: int) -> List:
        batch = list(islice(self.dataset, batch_size))
        if not batch or batch_size != len(batch):
            raise StopIteration
        return batch


@FILE_READERS.register_module(AlgorithmType.SFT.value)
class SFTDataReader(BufferReader):
    """Reader for SFT file data."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.split = meta.split
        subset_name = meta.subset_name
        self.prompt_type = meta.format.prompt_type
        self.messages_key = meta.format.messages_key
        self.prompt_key = meta.format.prompt_key
        self.response_key = meta.format.response_key
        self.read_batch_size = config.read_batch_size
        self.dataset = _HFBatchReader(
            load_dataset(meta.path, name=subset_name, split=self.split)
        )  # TODO: support resume
        self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        samples = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        if self.prompt_type == PromptType.MESSAGES:
            for sample in samples:
                messages = sample[self.messages_key]
                tokens = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, return_tensors="pt"
                )[0]
                prompt_tokens = self.tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, return_tensors="pt"
                )[0]
                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)

        elif self.prompt_type == PromptType.CHATPAIR:
            for sample in samples:
                prompt_messages = sample[self.prompt_key]
                response_messages = sample[self.response_key]
                if not isinstance(prompt_messages, list):
                    prompt_messages = [prompt_messages]
                if not isinstance(response_messages, list):
                    response_messages = [response_messages]
                full_messages = prompt_messages + response_messages

                tokens = self.tokenizer.apply_chat_template(
                    full_messages, add_generation_prompt=False, return_tensors="pt"
                )[0]

                prompt_tokens = self.tokenizer.apply_chat_template(
                    prompt_messages, add_generation_prompt=True, return_tensors="pt"
                )[0]

                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)

        elif self.prompt_type == PromptType.PLAINTEXT:
            # TODO: support HF format without chat template
            for sample in samples:
                prompt = sample[self.prompt_key]
                response = sample[self.response_key]
                tokens = self.tokenizer(prompt + response, return_tensors="pt")["input_ids"][0]
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)
        else:
            raise ValueError(f"Unknown data format: {self.prompt_type}")
        return exp_list


@FILE_READERS.register_module(AlgorithmType.DPO.value)
class DPODataReader(BufferReader):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.split = meta.split
        subset_name = meta.subset_name
        self.prompt_type = meta.format.prompt_type
        self.prompt_key = meta.format.prompt_key
        self.chosen_key = meta.format.chosen_key
        self.rejected_key = meta.format.rejected_key
        self.read_batch_size = config.read_batch_size
        self.dataset = _HFBatchReader(
            load_dataset(meta.path, name=subset_name, split=self.split)
        )  # TODO: support resume
        self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)

    def _get_assistant_message(self, item) -> dict:
        if isinstance(item, List):
            item = item[0]
        if isinstance(item, str):
            return {"role": "assistant", "content": item}
        else:
            return item

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        batch_data = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in batch_data:
            prompt = sample[self.prompt_key]
            chosen = sample[self.chosen_key]
            rejected = sample[self.rejected_key]

            if self.prompt_type == PromptType.MESSAGES:
                prompt_messages = prompt

            elif self.prompt_type == PromptType.PLAINTEXT:
                prompt_messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            else:
                raise ValueError(f"Unknown prompt type: {self.prompt_type}")
            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, return_tensors="pt"
            )[0]
            prompt_length = len(prompt_tokens)
            messages_with_chosen = prompt_messages + [self._get_assistant_message(chosen)]
            chosen_tokens = self.tokenizer.apply_chat_template(
                messages_with_chosen,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0][prompt_length:]
            messages_with_rejected = prompt_messages + [self._get_assistant_message(rejected)]
            rejected_tokens = self.tokenizer.apply_chat_template(
                messages_with_rejected,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0][prompt_length:]
            experience = Experience(
                tokens=prompt_tokens,
                prompt_length=len(prompt_tokens),
                chosen=chosen_tokens,
                rejected=rejected_tokens,
            )
            exp_list.append(experience)
        return exp_list


@FILE_READERS.register_module("rollout")
class RolloutDataReader(BufferReader):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.meta = meta
        self.name = meta.name
        self.split = meta.split
        subset_name = meta.subset_name
        # disable datasets caching to avoid reuse old-version dataset
        self.epoch = 0
        datasets.disable_caching()
        self.dataset = _HFBatchReader(load_dataset(meta.path, name=subset_name, split=self.split))
        if self.meta.index > 0:
            # offset the dataset to the correct index
            self.dataset.read_batch(self.meta.index)
        self.read_batch_size = config.batch_size
        self.prompt_key = meta.format.prompt_key
        self.response_key = meta.format.response_key
        self.workflow_key = meta.format.workflow_key
        self.reward_fn_key = meta.format.reward_fn_key

        self.task_type = meta.task_type
        self.default_workflow_cls = WORKFLOWS.get(meta.default_workflow_type)  # type: ignore
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(meta.default_reward_fn_type)  # type: ignore
        self.total_epochs = meta.total_epochs if self.task_type == TaskType.EXPLORE else 1

    def __len__(self):
        return len(self.dataset)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        try:
            samples = self.dataset.read_batch(batch_size)
        except StopIteration:
            self.epoch += 1
            if self.epoch >= self.total_epochs:
                raise StopIteration
        for sample in samples:
            workflow_class = (
                WORKFLOWS.get(sample[self.workflow_key])
                if self.workflow_key in sample
                else self.default_workflow_cls
            )
            reward_fn = (
                REWARD_FUNCTIONS.get(sample[self.reward_fn_key])
                if self.reward_fn_key in sample
                else self.default_reward_fn_cls
            )
            assert (
                workflow_class is not None
            ), "`default_workflow_type` or `workflow_key` is required"
            task = Task(
                workflow=workflow_class,
                format_args=self.meta.format,
                rollout_args=self.meta.rollout_args,
                workflow_args=self.meta.workflow_args,
                is_eval=self.meta.task_type == TaskType.EVAL,
                reward_fn=reward_fn,
                raw_task=sample,
            )
            tasks.append(task)
        return tasks
