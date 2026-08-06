# Copyright 2026 LongLive HSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD RainFusionAttention execution backend for inference-only HSA."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


_IMPORT_ERROR: Exception | None = None
try:
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
        rain_fusion_attention as _rain_fusion_attention,
    )
except Exception as error:  # pragma: no cover - depends on the NPU runtime
    _rain_fusion_attention = None
    _IMPORT_ERROR = error


def mindiesd_available() -> bool:
    return _rain_fusion_attention is not None


def mindiesd_unavailable_reason() -> str:
    if _IMPORT_ERROR is None:
        return "available"
    return f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"


@lru_cache(maxsize=128)
def _cached_selection_counts(
    device_type: str,
    device_index: int | None,
    q_blocks: int,
    heads: int,
    selected_blocks: int,
) -> torch.Tensor:
    return torch.full(
        (q_blocks, heads),
        selected_blocks,
        dtype=torch.int64,
        device=torch.device(device_type, device_index),
    )


def _prepare_mindiesd_lut(
    block_lut: torch.Tensor, k_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_lut.ndim != 4 or block_lut.shape[0] != 1:
        raise ValueError("MindIE-SD block_lut must have shape [1, heads, q_blocks, selected]")
    if block_lut.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"MindIE-SD block_lut must use int32 or int64 indices, got {block_lut.dtype}"
        )
    selected_blocks = block_lut.shape[-1]
    if selected_blocks <= 0 or selected_blocks > k_blocks:
        raise ValueError(
            f"block_lut selects {selected_blocks} blocks from only {k_blocks} KV blocks"
        )
    # RainFusionAttention defines the last dimension as maxKvBlockNum, the
    # maximum number selected by any row. LongLive selects a fixed count, so a
    # compact tensor is valid and avoids allocating/padding to the full KV width.
    select_idx = (
        block_lut[0]
        .permute(1, 0, 2)
        .to(dtype=torch.int64)
        .contiguous()
    )
    select_num_idx = _cached_selection_counts(
        select_idx.device.type,
        select_idx.device.index,
        select_idx.shape[0],
        select_idx.shape[1],
        selected_blocks,
    )
    return select_idx, select_num_idx


def mindiesd_sparse_attention_blhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Run the fused 128x128 RainFusionAttention operator on BLHD tensors.

    MindIE-SD consumes one LUT shared with the batch dimension, so LongLive's
    maintained batch-one inference path is required. TND layout keeps the
    existing BLHD tensors as views and avoids full Q/K/V transposes.
    """
    if _rain_fusion_attention is None:
        raise RuntimeError(f"MindIE-SD is unavailable: {mindiesd_unavailable_reason()}")
    if torch.is_grad_enabled():
        raise RuntimeError("MindIE-SD RainFusionAttention is forward-only")
    if q.device.type != "npu":
        raise RuntimeError(f"MindIE-SD RainFusionAttention requires NPU tensors, got {q.device}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("MindIE-SD HSA expects compatible BLHD q/k/v tensors")
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("MindIE-SD HSA currently requires batch size 1")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("MindIE-SD HSA requires matching q/k head dimensions")
    if q.device != k.device or q.device != v.device or block_lut.device != q.device:
        raise ValueError("MindIE-SD HSA requires q/k/v and block_lut on the same NPU device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("MindIE-SD HSA requires q/k/v to use the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"MindIE-SD RainFusionAttention supports FP16/BF16, got {q.dtype}"
        )
    if q.shape[1] % 128 or k.shape[1] % 128:
        raise ValueError("MindIE-SD HSA requires q and k token lengths divisible by 128")

    q_blocks = q.shape[1] // 128
    k_blocks = k.shape[1] // 128
    expected_prefix = (1, q.shape[2], q_blocks)
    if block_lut.ndim != 4 or tuple(block_lut.shape[:3]) != expected_prefix:
        raise ValueError(
            "block_lut must have shape [1, heads, q_blocks, selected], "
            f"expected prefix {expected_prefix}, got {tuple(block_lut.shape)}"
        )
    if block_lut.shape[-1] <= 0 or block_lut.shape[-1] > k_blocks:
        raise ValueError("block_lut selected-block count is outside the KV block range")

    select_idx, select_num_idx = _prepare_mindiesd_lut(block_lut, k_blocks)
    attention_scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    output = _rain_fusion_attention(
        q.squeeze(0).contiguous(),
        k.squeeze(0).contiguous(),
        v.squeeze(0).contiguous(),
        scale=attention_scale,
        head_num=q.shape[2],
        input_layout="TND",
        select_idx=select_idx,
        select_num_idx=select_num_idx,
        blockshape=[128, 128],
        actual_seq_lengths=[q.shape[1]],
        actual_seq_lengths_kv=[k.shape[1]],
        inner_precise=0,
    )
    return output.unsqueeze(0)


__all__ = [
    "mindiesd_available",
    "mindiesd_sparse_attention_blhd",
    "mindiesd_unavailable_reason",
]
