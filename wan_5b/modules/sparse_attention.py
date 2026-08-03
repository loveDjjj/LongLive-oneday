# Copyright 2026 LongLive HSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Chunk-Aware Growth and Hierarchical Sparse Attention.

The routing policy follows Light Forcing's two-stage design while keeping the
execution backend portable across CUDA and Ascend NPU. Historical frames are
selected first, then token blocks are ranked inside those frames. The current
chunk can be kept dense independently of the historical sparsity budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseAttentionConfig:
    enabled: bool = False
    sparsity: float = 0.85
    sparsity_base: float = 0.95
    block_q: int = 40
    block_k: int = 40
    keep_frames: int = 6
    keep_sink: int = 1
    keep_near: int = 2
    dense_current: bool = True
    min_sparse_history_frames: int = 1
    query_block_batch: int = 1
    softmax_scale: float | None = None
    num_output_frames: int | None = None
    num_frame_per_block: int = 1
    local_attn_size: int = -1
    sparsity_list: tuple[float, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SparseAttentionConfig":
        if not value:
            return cls()
        raw = dict(value)
        aliases = {
            "BLKQ": "block_q",
            "BLKK": "block_k",
            "backend": None,
        }
        normalized: dict[str, Any] = {}
        for key, item in raw.items():
            key = aliases.get(key, key)
            if key is not None and key in cls.__dataclass_fields__:
                normalized[key] = item
        if "sparsity_list" in normalized:
            normalized["sparsity_list"] = tuple(float(x) for x in normalized["sparsity_list"])
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("sparsity", "sparsity_base"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")
        if self.block_q <= 0 or self.block_k <= 0:
            raise ValueError("block_q and block_k must be positive.")
        if min(self.keep_frames, self.keep_sink, self.keep_near) < 0:
            raise ValueError("frame keep counts must be non-negative.")
        if self.enabled and self.keep_frames == 0:
            raise ValueError("keep_frames must be positive when HSA is enabled.")
        if self.enabled and not self.dense_current:
            raise ValueError(
                "The portable HSA backend requires dense_current=true; "
                "sparse current-chunk routing is not implemented."
            )
        if self.keep_sink + self.keep_near > self.keep_frames:
            raise ValueError("keep_sink + keep_near must not exceed keep_frames.")
        if self.query_block_batch <= 0:
            raise ValueError("query_block_batch must be positive.")


def calculate_chunk_sparsities(
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sparse_config: Mapping[str, Any] | SparseAttentionConfig | None,
) -> list[float]:
    """Allocate CAG sparsity while preserving the requested average FLOPs.

    Earlier chunks receive a larger attention budget. The first chunk is dense,
    matching the Light Forcing initialization policy.
    """
    config = (
        sparse_config
        if isinstance(sparse_config, SparseAttentionConfig)
        else SparseAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled or num_output_frames <= 0:
        return []
    if num_frame_per_block <= 0:
        raise ValueError("num_frame_per_block must be positive.")

    chunk_frame_counts = list(
        range(2 * num_frame_per_block, num_output_frames + 1, num_frame_per_block)
    )
    if not chunk_frame_counts:
        return [0.0]
    kv_lengths = [
        count if local_attn_size == -1 else min(count, local_attn_size)
        for count in chunk_frame_counts
    ]
    alphas = [1.0 / math.sqrt(count) for count in chunk_frame_counts]
    target_flops = sum((1.0 - config.sparsity) * length for length in kv_lengths)
    base_flops = sum((1.0 - config.sparsity_base) * length for length in kv_lengths)
    weighted_flops = sum(alpha * length for alpha, length in zip(alphas, kv_lengths))
    if weighted_flops == 0:
        tail = [config.sparsity_base] * len(alphas)
    else:
        beta = (target_flops - base_flops) / weighted_flops
        tail = [config.sparsity_base - alpha * beta for alpha in alphas]
    return [0.0] + [min(0.999, max(0.0, value)) for value in tail]


def resolve_chunk_sparsity(config: SparseAttentionConfig, chunk_id: int) -> float:
    if chunk_id <= 0:
        return 0.0
    if config.sparsity_list:
        return config.sparsity_list[min(chunk_id, len(config.sparsity_list) - 1)]
    if config.num_output_frames:
        schedule = calculate_chunk_sparsities(
            config.num_output_frames,
            config.num_frame_per_block,
            config.local_attn_size,
            config,
        )
        return schedule[min(chunk_id, len(schedule) - 1)]
    return config.sparsity


def _dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()


def _required_history_frames(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    history_frames: int,
    blocks_per_frame: int,
    config: SparseAttentionConfig,
) -> torch.Tensor:
    """Return selected historical frame ids as [B, H, QBlocks, FKeep]."""
    b, q_count, heads, _ = q_blocks.shape
    keep_count = min(config.keep_frames, history_frames)
    if keep_count >= history_frames:
        return torch.arange(history_frames, device=q_blocks.device).view(1, 1, 1, -1).expand(
            b, heads, q_count, -1
        )

    sink_count = min(config.keep_sink, keep_count)
    near_count = min(config.keep_near, keep_count - sink_count)
    middle_keep = keep_count - sink_count - near_count
    fixed_parts = []
    if sink_count:
        fixed_parts.append(torch.arange(sink_count, device=q_blocks.device))
    if near_count:
        fixed_parts.append(torch.arange(history_frames - near_count, history_frames, device=q_blocks.device))

    selected = []
    if fixed_parts:
        fixed = torch.cat(fixed_parts).view(1, 1, 1, -1).expand(b, heads, q_count, -1)
        selected.append(fixed)
    if middle_keep:
        middle_start = sink_count
        middle_end = history_frames - near_count
        history = k_blocks[:, : history_frames * blocks_per_frame]
        frame_keys = history.reshape(
            b, history_frames, blocks_per_frame, heads, -1
        ).mean(dim=2)
        middle_keys = frame_keys[:, middle_start:middle_end]
        scores = torch.einsum("bmhd,bfhd->bhmf", q_blocks.float(), middle_keys.float())
        middle_ids = torch.topk(scores, middle_keep, dim=-1, sorted=False).indices + middle_start
        selected.append(middle_ids)
    return torch.cat(selected, dim=-1)


def _history_block_indices(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    history_frames: int,
    blocks_per_frame: int,
    history_keep_blocks: int,
    config: SparseAttentionConfig,
) -> torch.Tensor:
    """Select historical key blocks with HSA's frame then block hierarchy."""
    b, q_count, heads, _ = q_blocks.shape
    history_block_count = history_frames * blocks_per_frame
    if history_keep_blocks >= history_block_count:
        return torch.arange(history_block_count, device=q_blocks.device).view(1, 1, 1, -1).expand(
            b, heads, q_count, -1
        )

    with torch.no_grad():
        frame_ids = _required_history_frames(
            q_blocks.detach(), k_blocks.detach(), history_frames, blocks_per_frame, config
        )
        offsets = torch.arange(blocks_per_frame, device=q_blocks.device)
        candidates = (
            frame_ids.unsqueeze(-1) * blocks_per_frame + offsets.view(1, 1, 1, 1, -1)
        ).flatten(-2)

        keys = k_blocks[:, :history_block_count].permute(0, 2, 1, 3)
        keys = keys.unsqueeze(2).expand(-1, -1, q_count, -1, -1)
        candidate_keys = torch.gather(
            keys,
            3,
            candidates.unsqueeze(-1).expand(-1, -1, -1, -1, keys.shape[-1]),
        )
        scores = torch.einsum(
            "bmhd,bhmtd->bhmt",
            q_blocks.detach().float(),
            candidate_keys.float(),
        )
        keep = min(history_keep_blocks, candidates.shape[-1])
        chosen = torch.topk(scores, keep, dim=-1, sorted=False).indices
        return torch.gather(candidates, -1, chosen)


def _gather_query_blocks(
    blocks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather [B,H,N,Block,D] with [B,H,Q,Selected] indices."""
    query_count = indices.shape[2]
    expanded = blocks.unsqueeze(2).expand(-1, -1, query_count, -1, -1, -1)
    return torch.gather(
        expanded,
        3,
        indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, -1, blocks.shape[-2], blocks.shape[-1]
        ),
    )


def hierarchical_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | SparseAttentionConfig | None,
) -> torch.Tensor:
    """Apply HSA to BLHD tensors using a portable block-gather backend."""
    config = (
        sparse_config
        if isinstance(sparse_config, SparseAttentionConfig)
        else SparseAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled:
        return _dense_attention(q, k, v, config.softmax_scale)
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("HSA expects q/k/v in BLHD layout with compatible shapes.")
    if (
        frame_seq <= 0
        or frame_seq % config.block_q != 0
        or frame_seq % config.block_k != 0
    ):
        raise ValueError(
            f"frame_seq ({frame_seq}) must be divisible by block_q/block_k "
            f"({config.block_q}/{config.block_k})."
        )
    if q.shape[1] % frame_seq or k.shape[1] % frame_seq:
        raise ValueError("q and k token lengths must contain complete latent frames.")

    history_tokens = k.shape[1] - q.shape[1]
    if history_tokens <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)
    history_frames = history_tokens // frame_seq
    if history_frames < config.min_sparse_history_frames:
        return _dense_attention(q, k, v, config.softmax_scale)

    sparsity = resolve_chunk_sparsity(config, chunk_id)
    if sparsity <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)

    block_q, block_k = config.block_q, config.block_k
    q_work = q
    k_work, v_work = k, v

    b, _, heads, dim = q_work.shape
    q_count = q_work.shape[1] // block_q
    k_count = k_work.shape[1] // block_k
    history_block_count = history_tokens // block_k
    current_block_ids = torch.arange(history_block_count, k_count, device=q.device)
    history_keep = max(1, math.ceil((1.0 - sparsity) * history_block_count))

    q_blocks = q_work.reshape(b, q_count, block_q, heads, dim).mean(dim=2)
    k_block_means = k_work.reshape(b, k_count, block_k, heads, dim).mean(dim=2)
    history_ids = _history_block_indices(
        q_blocks,
        k_block_means,
        history_frames,
        frame_seq // block_k,
        history_keep,
        config,
    )

    k_blocks = k_work.reshape(b, k_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    v_blocks = v_work.reshape(b, k_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    q_heads = q_work.permute(0, 2, 1, 3)
    outputs = []
    group = config.query_block_batch
    for start in range(0, q_count, group):
        end = min(start + group, q_count)
        group_history = history_ids[:, :, start:end]
        if config.dense_current:
            current = current_block_ids.view(1, 1, 1, -1).expand(
                b, heads, end - start, -1
            )
            group_history = torch.cat([group_history, current], dim=-1)
        selected = torch.sort(group_history, dim=-1).values
        selected_k = _gather_query_blocks(k_blocks, selected).flatten(3, 4)
        selected_v = _gather_query_blocks(v_blocks, selected).flatten(3, 4)

        q_start = start * block_q
        q_end = end * block_q
        q_group = q_heads[:, :, q_start:q_end].reshape(
            b, heads, end - start, block_q, dim
        )
        batch_heads_queries = b * heads * (end - start)
        out = F.scaled_dot_product_attention(
            q_group.reshape(batch_heads_queries, block_q, dim),
            selected_k.reshape(batch_heads_queries, selected_k.shape[-2], dim),
            selected_v.reshape(batch_heads_queries, selected_v.shape[-2], dim),
            scale=config.softmax_scale,
        )
        outputs.append(out.reshape(b, heads, (end - start) * block_q, dim))
    result = torch.cat(outputs, dim=2)[:, :, : q.shape[1]]
    return result.permute(0, 2, 1, 3).contiguous()


def with_cag_schedule(
    sparse_config: Mapping[str, Any] | None,
    *,
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
) -> dict[str, Any]:
    """Return a config copy with a concrete CAG schedule."""
    output = dict(sparse_config or {})
    parsed = SparseAttentionConfig.from_mapping(output)
    output["num_output_frames"] = num_output_frames
    output["num_frame_per_block"] = num_frame_per_block
    output["local_attn_size"] = local_attn_size
    output["sparsity_list"] = calculate_chunk_sparsities(
        num_output_frames, num_frame_per_block, local_attn_size, parsed
    )
    return output


__all__ = [
    "SparseAttentionConfig",
    "calculate_chunk_sparsities",
    "hierarchical_sparse_attention",
    "resolve_chunk_sparsity",
    "with_cag_schedule",
]
