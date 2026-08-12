# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""Method selection and shared CAG scheduling for sparse attention."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .hsa_attention import HSAAttentionConfig, hsa_cag_attention
from .sla_attention import SLAAttentionConfig, sla_cag_attention


SPARSE_METHODS = ("hsa_cag", "sla_cag")


def sparse_method(value: Mapping[str, Any] | None) -> str:
    config = dict(value or {})
    if not config.get("enabled", False):
        return "dense"
    method = str(config.get("method", "")).strip()
    if not method:
        method = "hsa_cag" if "keep_frames" in config else "sla_cag"
    if method not in SPARSE_METHODS:
        raise ValueError(
            f"unsupported sparse attention method={method!r}; choose one of {SPARSE_METHODS}"
        )
    return method


def parse_sparse_config(value: Mapping[str, Any] | None):
    method = sparse_method(value)
    if method == "dense":
        return SLAAttentionConfig()
    if method == "hsa_cag":
        return HSAAttentionConfig.from_mapping(value)
    return SLAAttentionConfig.from_mapping(value)


def calculate_chunk_sparsities(
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sparse_config: Mapping[str, Any] | None,
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
    attention_cache=None,
):
    method = sparse_method(sparse_config)
    if method == "hsa_cag":
        return hsa_cag_attention(
            q, k, v, frame_seq=frame_seq, chunk_id=chunk_id,
            sparse_config=sparse_config, attention_cache=attention_cache,
        )
    if method == "sla_cag":
        return sla_cag_attention(
            q, k, v, frame_seq=frame_seq, chunk_id=chunk_id,
            sparse_config=sparse_config, linear_projection=linear_projection,
            attention_cache=attention_cache,
        )
    raise AssertionError(f"unexpected sparse method: {method}")


__all__ = [
    "SPARSE_METHODS", "calculate_chunk_sparsities", "parse_sparse_config",
    "sparse_attention", "sparse_method", "with_cag_schedule",
]
