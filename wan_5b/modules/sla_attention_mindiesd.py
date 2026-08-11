# Copyright 2026 LongLive SLA contributors.
# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD RainFusionAttention execution backend for inference-only SLA sparse branch."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


_IMPORT_ERROR: Exception | None = None
_BSA_IMPORT_ERROR: Exception | None = None
try:
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
        rain_fusion_attention as _rain_fusion_attention,
    )
except Exception as error:  # pragma: no cover - depends on the NPU runtime
    _rain_fusion_attention = None
    _IMPORT_ERROR = error

try:
    from mindiesd.layers._custom_ops import (
        block_sparse_attention as _block_sparse_attention,
    )
except Exception as error:  # pragma: no cover - depends on the NPU runtime
    _block_sparse_attention = None
    _BSA_IMPORT_ERROR = error


def mindiesd_available() -> bool:
    return _rain_fusion_attention is not None


def mindiesd_unavailable_reason() -> str:
    if _IMPORT_ERROR is None:
        return "available"
    return f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"


def mindiesd_bsa_available() -> bool:
    return _block_sparse_attention is not None and hasattr(
        torch.ops.mindiesd, "block_sparse_attention"
    )


def mindiesd_bsa_unavailable_reason() -> str:
    if _BSA_IMPORT_ERROR is None:
        if _block_sparse_attention is None:
            return "MindIE-SD block_sparse_attention wrapper is unavailable"
        if not hasattr(torch.ops.mindiesd, "block_sparse_attention"):
            return "MindIE-SD block_sparse_attention operator is not registered"
        return "available"
    return f"{type(_BSA_IMPORT_ERROR).__name__}: {_BSA_IMPORT_ERROR}"


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


def _prepare_mindiesd_bsa_mask(
    block_lut: torch.Tensor, k_blocks: int
) -> torch.Tensor:
    if block_lut.ndim != 4:
        raise ValueError("MindIE-SD BSA LUT must have shape [batch, heads, q_blocks, selected]")
    if block_lut.shape[-1] <= 0 or block_lut.shape[-1] > k_blocks:
        raise ValueError("MindIE-SD BSA selected-block count is outside the KV block range")
    mask = torch.zeros(
        (*block_lut.shape[:3], k_blocks),
        dtype=torch.int8,
        device=block_lut.device,
    )
    return mask.scatter_(-1, block_lut.long(), 1)


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
        raise ValueError("MindIE-SD SLA sparse branch expects compatible BLHD q/k/v tensors")
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("MindIE-SD SLA sparse branch currently requires batch size 1")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("MindIE-SD SLA sparse branch requires matching q/k head dimensions")
    if q.device != k.device or q.device != v.device or block_lut.device != q.device:
        raise ValueError("MindIE-SD SLA sparse branch requires q/k/v and block_lut on the same NPU device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("MindIE-SD SLA sparse branch requires q/k/v to use the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"MindIE-SD RainFusionAttention supports FP16/BF16, got {q.dtype}"
        )
    if q.shape[1] % 128 or k.shape[1] % 128:
        raise ValueError("MindIE-SD SLA sparse branch requires q and k token lengths divisible by 128")

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


def mindiesd_bsa_sparse_attention_blhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Run MindIE-SD BlockSparseAttention directly in rectangular TND layout."""
    if _block_sparse_attention is None:
        raise RuntimeError(
            "MindIE-SD BSA is unavailable: " + mindiesd_bsa_unavailable_reason()
        )
    if torch.is_grad_enabled():
        raise RuntimeError("MindIE-SD BlockSparseAttention is forward-only")
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("MindIE-SD BSA expects batch-one compatible BLHD q/k/v")
    if q.device.type != "npu" or q.device != k.device or q.device != v.device:
        raise ValueError("MindIE-SD BSA requires q/k/v on the same NPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or not (
        q.dtype == k.dtype == v.dtype
    ):
        raise TypeError("MindIE-SD BSA requires matching BF16/FP16 q/k/v")
    if q.shape[2:] != k.shape[2:] or q.shape[1] % 128 or k.shape[1] % 128:
        raise ValueError("MindIE-SD BSA requires matching heads and 128-aligned sequence lengths")

    q_blocks = q.shape[1] // 128
    k_blocks = k.shape[1] // 128
    if tuple(block_lut.shape[:3]) != (1, q.shape[2], q_blocks):
        raise ValueError("MindIE-SD BSA LUT shape does not match q")
    block_mask = _prepare_mindiesd_bsa_mask(block_lut, k_blocks)
    attention_scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    output, _ = _block_sparse_attention(
        query=q.squeeze(0).contiguous(),
        key=k.squeeze(0).contiguous(),
        value=v.squeeze(0).contiguous(),
        block_sparse_mask=block_mask,
        block_shape=[128, 128],
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=q.shape[2],
        scale_value=attention_scale,
        inner_precise=0,
        actual_seq_lengths=[q.shape[1]],
        actual_seq_lengths_kv=[k.shape[1]],
        softmax_lse_flag=0,
    )
    return output.unsqueeze(0)


__all__ = [
    "mindiesd_bsa_available",
    "mindiesd_bsa_sparse_attention_blhd",
    "mindiesd_bsa_unavailable_reason",
    "mindiesd_available",
    "mindiesd_sparse_attention_blhd",
    "mindiesd_unavailable_reason",
]
