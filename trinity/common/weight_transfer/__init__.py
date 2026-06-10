# -*- coding: utf-8 -*-
"""Bucketed model weight transfer between trainer and rollout workers.

Public API:

* :class:`ModelWeightSender` — trainer-side sender with double-buffered NCCL broadcast.
* :class:`ModelWeightReceiver` — rollout-side receiver with async and sync iterators.
* :class:`TensorMeta` — per-chunk metadata carried alongside bucket data.
"""
from trinity.common.weight_transfer.core import TensorMeta
from trinity.common.weight_transfer.nccl import (
    ModelWeightMetadataReceiver,
    ModelWeightReceiver,
    ModelWeightSender,
)

__all__ = [
    "ModelWeightSender",
    "ModelWeightReceiver",
    "ModelWeightMetadataReceiver",
    "TensorMeta",
]
