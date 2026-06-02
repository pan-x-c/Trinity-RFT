"""Utils for ccompatibility issues with verl."""

from logging import Logger
from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel
from verl import DataProto

from trinity.common.experience import (
    Experience,
    gather_action_masks,
    gather_attention_masks,
    gather_response_attrs,
    gather_token_ids,
    split_dpo_experience_to_single_turn,
)


def _gather_routed_experts(
    experiences: List[Experience], max_prompt_length: int, max_response_length: int
) -> torch.Tensor:
    """Pad routed experts to the full left-padded sequence layout expected by verl."""
    batch_size = len(experiences)
    total_length = max_prompt_length + max_response_length
    routed_experts = experiences[0].routed_experts
    assert routed_experts is not None, "No routed_experts provided."
    _, layer_num, topk_num = routed_experts.shape
    batch_routed_experts = torch.zeros(
        batch_size,
        total_length,
        layer_num,
        topk_num,
        dtype=routed_experts.dtype,
    )

    for idx, exp in enumerate(experiences):
        exp_routed_experts = exp.routed_experts
        assert exp_routed_experts is not None, "No routed_experts provided."
        if exp_routed_experts.ndim != 3:
            raise ValueError(
                "Experience.routed_experts must have shape [seq_length - 1, layer_num, topk]."
            )
        if exp_routed_experts.shape[1:] != (layer_num, topk_num):
            raise ValueError("routed_experts shape is inconsistent across experiences.")

        start_pos = max_prompt_length - exp.prompt_length
        end_pos = min(start_pos + exp_routed_experts.shape[0], total_length)
        batch_routed_experts[idx, start_pos:end_pos] = exp_routed_experts[: end_pos - start_pos]

    return batch_routed_experts


def to_data_proto(  # noqa: C901
    experiences: List[Experience], pad_token_id: int, model: PreTrainedModel, logger: Logger
) -> DataProto:
    """Convert List[Experience] to verl DataProto."""
    assert len(experiences) > 0, "No experiences provided."
    if experiences[0].experience_type == "dpo":
        experiences = split_dpo_experience_to_single_turn(experiences)
    max_prompt_length = max([exp.prompt_length for exp in experiences])
    max_response_length = max([len(exp.tokens) - exp.prompt_length for exp in experiences])  # type: ignore

    attention_mask = gather_attention_masks(
        experiences, max_prompt_length, max_response_length
    ).long()
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    tokens = gather_token_ids(
        experiences, max_prompt_length, max_response_length, pad_token_id
    ).long()
    batch_dict = {
        "uid": np.array([exp.eid.tid for exp in experiences]),
        "unique_ids": np.array([exp.eid.uid for exp in experiences]),
        "position_ids": position_ids,
        "input_ids": tokens,
        "responses": tokens[:, max_prompt_length:],
        "attention_mask": attention_mask,
        "response_mask": gather_action_masks(experiences, max_response_length),
    }

    have_reward = all(exp.reward is not None for exp in experiences)
    have_token_level_reward = all(exp.token_level_reward is not None for exp in experiences)
    if have_reward or have_token_level_reward:
        assert all(exp.logprobs is not None for exp in experiences), "No logprobs provided."
        if have_token_level_reward:
            if have_reward:
                logger.warning(
                    "Both experiences.rewards and experiences.token_level_rewards are provided. "
                    "Using experiences.token_level_rewards."
                )
            token_level_rewards = gather_response_attrs(
                experiences, "token_level_reward", max_response_length
            )
        else:
            token_level_rewards = torch.zeros(attention_mask.shape, dtype=torch.float32)
            eos_mask_idx = cumsum.argmax(dim=-1)
            token_level_rewards[torch.arange(len(experiences)), eos_mask_idx] = torch.tensor(
                [exp.reward for exp in experiences],
                dtype=torch.float32,
            )
            token_level_rewards = token_level_rewards[:, max_prompt_length:]
        batch_dict.update(
            {
                "token_level_scores": token_level_rewards,
                "rollout_log_probs": gather_response_attrs(
                    experiences, "logprobs", max_response_length
                ),
            }
        )

    for attr in ["advantages", "returns", "teacher_logprobs"]:
        if all(getattr(exp, attr, None) is not None for exp in experiences):
            batch_dict[attr] = gather_response_attrs(experiences, attr, max_response_length)

    if any(exp.routed_experts is not None for exp in experiences):
        if not all(exp.routed_experts is not None for exp in experiences):
            raise ValueError("routed_experts are not consistent across experiences.")
        batch_dict["routed_experts"] = _gather_routed_experts(
            experiences, max_prompt_length, max_response_length
        )

    if hasattr(model, "get_rope_index"):
        # used for multi-modal model
        import inspect

        # Adapted from verl/experimental/agent_loop/agent_loop.py
        position_ids_list, multi_modal_inputs = [], []
        for idx, exp in enumerate(experiences):
            mm_inputs = exp.multi_modal_inputs or {}
            input_ids = batch_dict["input_ids"][idx].unsqueeze(0)
            attention_mask = batch_dict["attention_mask"][idx].unsqueeze(0)

            get_rope_index_sig = inspect.signature(model.get_rope_index)
            get_rope_index_kwargs = {}
            for key in get_rope_index_sig.parameters:
                if key in {"self", "input_ids", "attention_mask", "kwargs"}:
                    continue
                elif key == "mm_token_type_ids":
                    pad_data = torch.zeros_like(input_ids)
                    if key in mm_inputs:
                        data = mm_inputs.pop(key)
                        start = max_prompt_length - exp.prompt_length
                        end = start + data.size(1)
                        pad_data[:, start:end] = data
                    get_rope_index_kwargs[key] = pad_data
                else:
                    get_rope_index_kwargs[key] = mm_inputs.get(key, None)

            vision_position_ids, _ = model.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **get_rope_index_kwargs,
            )  # (3, 1, seq_len)
            vision_position_ids = vision_position_ids.squeeze(1)  # (3, seq_len)

            text_position_ids = batch_dict["position_ids"][idx].unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat(
                (text_position_ids, vision_position_ids), dim=0
            )  # (4, seq_length)
            position_ids_list.append(position_ids)  # (4, seq_length)
            multi_modal_inputs.append(mm_inputs)

        batch_dict["position_ids"] = torch.stack(
            position_ids_list, dim=0
        ).long()  # (bs, 4, seq_length)
        batch_dict["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

    custom_fields_set = set(tuple(exp.custom_fields) for exp in experiences)
    if len(custom_fields_set) == 1:
        custom_fields = list(custom_fields_set)[0]
        for custom_field in custom_fields:
            batch_dict[custom_field.destination_field] = torch.tensor(
                [exp.info[custom_field.source_field] for exp in experiences],
                dtype=custom_field.data_type,
            )
    else:
        raise ValueError("Custom fields are not consistent across experiences.")
    meta_info = {
        "model_versions": np.array([exp.info.get("model_version", 0) for exp in experiences])
    }
    return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
