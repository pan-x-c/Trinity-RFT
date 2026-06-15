# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import logging

from trinity.common.models.vllm_patch.worker_patch import patch_vllm_prompt_logprobs


def _suppress_layerwise_reload_warnings() -> None:
    """Silence benign vLLM layerwise reload warnings during weight sync."""
    try:
        logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        if logger is not None:
            logger.setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - best-effort suppression
        pass


class WorkerExtension:
    def apply_patches(self):
        """Apply necessary patches to vLLM."""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        patch_vllm_prompt_logprobs(self.model_runner)
        _suppress_layerwise_reload_warnings()
