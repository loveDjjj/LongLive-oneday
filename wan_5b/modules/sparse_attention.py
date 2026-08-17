# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""Method selection and shared CAG scheduling for sparse attention."""

from __future__ import annotations

import math
from typing import Any, Mapping, TypeAlias

from .hsa_attention import HSAAttentionConfig, hsa_cag_attention
from .hsa_sla_attention import HSASLAAttentionConfig, hsa_sla_cag_attention
from .sla_attention import SLAAttentionConfig, sla_cag_attention
from .sparse_routing import SparseKVLayout, reset_sparse_routing_debug


SPARSE_METHODS = ("hsa_cag", "sla_cag", "hsa_sla_cag")
SPARSE_CACHE_KEYS = ("sparse_attention_cache", "sla_attention_cache")
ParsedSparseConfig: TypeAlias = (
    HSAAttentionConfig | SLAAttentionConfig | HSASLAAttentionConfig
)
SparseConfig: TypeAlias = Mapping[str, Any] | ParsedSparseConfig


def clear_sparse_attention_cache(kv_cache) -> None:
    """Remove per-video router/linear statistics from every transformer block."""
    for block_cache in kv_cache:
        for key in SPARSE_CACHE_KEYS:
            block_cache.pop(key, None)
    reset_sparse_routing_debug()


def sparse_method(value: SparseConfig | None) -> str:
    if isinstance(value, HSAAttentionConfig):
        return "hsa_cag" if value.enabled else "dense"
    if isinstance(value, HSASLAAttentionConfig):
        return "hsa_sla_cag" if value.enabled else "dense"
    if isinstance(value, SLAAttentionConfig):
        return "sla_cag" if value.enabled else "dense"
    config = dict(value or {})
    if not config.get("enabled", False):
        return "dense"
    method = str(config.get("method", "")).strip()
    if not method:
        raise ValueError("enabled sparse attention requires an explicit method.")
    if method not in SPARSE_METHODS:
        raise ValueError(
            f"unsupported sparse attention method={method!r}; choose one of {SPARSE_METHODS}"
        )
    return method


def parse_sparse_config(value: SparseConfig | None) -> ParsedSparseConfig:
    if isinstance(value, (HSAAttentionConfig, SLAAttentionConfig, HSASLAAttentionConfig)):
        return value
    method = sparse_method(value)
    if method == "dense":
        return SLAAttentionConfig()
    if method == "hsa_cag":
        return HSAAttentionConfig.from_mapping(value)
    if method == "hsa_sla_cag":
        return HSASLAAttentionConfig.from_mapping(value)
    return SLAAttentionConfig.from_mapping(value)


def calculate_chunk_sparsities(
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sparse_config: SparseConfig | None,
) -> list[float]:
    config = parse_sparse_config(sparse_config)
    if not config.enabled or num_output_frames <= 0:
        return []
    chunk_frames = list(
        range(2 * num_frame_per_block, num_output_frames + 1, num_frame_per_block)
    )
    if not chunk_frames:
        return [0.0]
    kv_lengths = [
        count if local_attn_size == -1 else min(count, local_attn_size)
        for count in chunk_frames
    ]
    alphas = [1.0 / math.sqrt(count) for count in chunk_frames]
    target = sum((1.0 - config.sparsity) * length for length in kv_lengths)
    base = sum((1.0 - config.sparsity_base) * length for length in kv_lengths)
    weighted = sum(alpha * length for alpha, length in zip(alphas, kv_lengths))
    beta = 0.0 if weighted == 0 else (target - base) / weighted
    return [0.0] + [
        min(0.999, max(0.0, config.sparsity_base - alpha * beta))
        for alpha in alphas
    ]


def with_cag_schedule(
    sparse_config: Mapping[str, Any] | None,
    *,
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
) -> dict[str, Any]:
    output = dict(sparse_config or {})
    method = sparse_method(output)
    if method != "dense":
        output["method"] = method
    output["num_output_frames"] = num_output_frames
    output["num_frame_per_block"] = num_frame_per_block
    output["local_attn_size"] = local_attn_size
    output["sparsity_list"] = calculate_chunk_sparsities(
        num_output_frames, num_frame_per_block, local_attn_size, output
    )
    return output


def sparse_attention(
    q,
    k,
    v,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config,
    linear_projection,
    kv_layout: SparseKVLayout | None = None,
    attention_cache=None,
):
    method = sparse_method(sparse_config)
    if method == "hsa_cag":
        return hsa_cag_attention(
            q, k, v, frame_seq=frame_seq, chunk_id=chunk_id,
            sparse_config=sparse_config, kv_layout=kv_layout,
            attention_cache=attention_cache,
        )
    if method == "sla_cag":
        return sla_cag_attention(
            q, k, v, frame_seq=frame_seq, chunk_id=chunk_id,
            sparse_config=sparse_config, linear_projection=linear_projection,
            kv_layout=kv_layout,
            attention_cache=attention_cache,
        )
    if method == "hsa_sla_cag":
        return hsa_sla_cag_attention(
            q, k, v, frame_seq=frame_seq, chunk_id=chunk_id,
            sparse_config=sparse_config, linear_projection=linear_projection,
            kv_layout=kv_layout,
            attention_cache=attention_cache,
        )
    raise AssertionError(f"unexpected sparse method: {method}")


__all__ = [
    "SPARSE_METHODS", "calculate_chunk_sparsities", "clear_sparse_attention_cache",
    "parse_sparse_config", "sparse_attention", "sparse_method", "with_cag_schedule",
]
