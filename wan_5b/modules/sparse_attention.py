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
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_ASCEND_BLHD_INFERENCE = os.environ.get("LONGLIVE_HSA_BLHD_INFERENCE", "1") == "1"
_ASCEND_VALIDATE_LUT = os.environ.get("LONGLIVE_HSA_VALIDATE_LUT", "0") == "1"


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


@lru_cache(maxsize=128)
def _cached_arange(
    device_type: str,
    device_index: int | None,
    start: int,
    end: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.arange(start, end, device=device)


@lru_cache(maxsize=64)
def _cached_frame_block_overlap(
    device_type: str,
    device_index: int | None,
    history_frames: int,
    frame_seq: int,
    block_k: int,
    history_block_count: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    frame_starts = torch.arange(history_frames, device=device) * frame_seq
    frame_ends = frame_starts + frame_seq
    block_starts = torch.arange(history_block_count, device=device) * block_k
    block_ends = block_starts + block_k
    return (
        (block_starts.unsqueeze(0) < frame_ends.unsqueeze(1))
        & (block_ends.unsqueeze(0) > frame_starts.unsqueeze(1))
    )


@dataclass(frozen=True)
class SparseAttentionConfig:
    enabled: bool = False
    backend: str = "portable"
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
        }
        normalized: dict[str, Any] = {}
        for key, item in raw.items():
            key = aliases.get(key, key)
            if key is not None and key in cls.__dataclass_fields__:
                normalized[key] = item
        if "sparsity_list" in normalized:
            normalized["sparsity_list"] = tuple(float(x) for x in normalized["sparsity_list"])
        if "backend" in normalized:
            backend_aliases = {
                "torch": "portable",
                "triton": "ascend_triton",
                "npu_triton": "ascend_triton",
                "mindie_sd": "mindiesd",
                "rainfusion": "mindiesd",
            }
            backend = str(normalized["backend"]).lower()
            normalized["backend"] = backend_aliases.get(backend, backend)
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.backend not in {"portable", "ascend_triton", "mindiesd", "auto"}:
            raise ValueError(
                "backend must be one of portable, ascend_triton, mindiesd, or auto; "
                f"got {self.backend}."
            )
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
                "HSA requires dense_current=true; "
                "sparse current-chunk routing is not implemented."
            )
        if self.keep_sink + self.keep_near > self.keep_frames:
            raise ValueError("keep_sink + keep_near must not exceed keep_frames.")
        if self.query_block_batch <= 0:
            raise ValueError("query_block_batch must be positive.")
        if self.enabled and self.backend == "mindiesd" and (
            self.block_q != 128 or self.block_k != 128
        ):
            raise ValueError(
                "MindIE-SD RainFusionAttention requires block_q=block_k=128."
            )


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
    history_k: torch.Tensor,
    history_frames: int,
    frame_seq: int,
    config: SparseAttentionConfig,
    history_frame_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected historical frame ids as [B, H, QBlocks, FKeep]."""
    b, q_count, heads, _ = q_blocks.shape
    keep_count = min(config.keep_frames, history_frames)
    device_type, device_index = _device_key(q_blocks.device)
    if keep_count >= history_frames:
        return _cached_arange(
            device_type, device_index, 0, history_frames
        ).view(1, 1, 1, -1).expand(b, heads, q_count, -1)

    sink_count = min(config.keep_sink, keep_count)
    near_count = min(config.keep_near, keep_count - sink_count)
    middle_keep = keep_count - sink_count - near_count
    fixed_parts = []
    if sink_count:
        fixed_parts.append(
            _cached_arange(device_type, device_index, 0, sink_count)
        )
    if near_count:
        fixed_parts.append(
            _cached_arange(
                device_type,
                device_index,
                history_frames - near_count,
                history_frames,
            )
        )

    selected = []
    if fixed_parts:
        fixed = torch.cat(fixed_parts).view(1, 1, 1, -1).expand(b, heads, q_count, -1)
        selected.append(fixed)
    if middle_keep:
        middle_start = sink_count
        middle_end = history_frames - near_count
        frame_keys = history_frame_keys
        if frame_keys is None:
            frame_keys = history_k.reshape(
                b, history_frames, frame_seq, heads, -1
            ).mean(dim=2)
        middle_keys = frame_keys[:, middle_start:middle_end]
        scores = torch.matmul(
            q_blocks.permute(0, 2, 1, 3).float(),
            middle_keys.permute(0, 2, 3, 1).float(),
        )
        middle_ids = torch.topk(scores, middle_keep, dim=-1, sorted=False).indices + middle_start
        selected.append(middle_ids)
    return torch.cat(selected, dim=-1)


def _history_block_indices(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    history_k: torch.Tensor,
    history_frames: int,
    frame_seq: int,
    block_k: int,
    history_keep_blocks: int,
    config: SparseAttentionConfig,
    history_frame_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select historical key blocks with HSA's frame then block hierarchy."""
    b, q_count, heads, _ = q_blocks.shape
    history_block_count = history_k.shape[1] // block_k
    if history_keep_blocks >= history_block_count:
        device_type, device_index = _device_key(q_blocks.device)
        return _cached_arange(
            device_type, device_index, 0, history_block_count
        ).view(1, 1, 1, -1).expand(b, heads, q_count, -1)

    with torch.no_grad():
        frame_ids = _required_history_frames(
            q_blocks.detach(), history_k.detach(), history_frames, frame_seq,
            config, history_frame_keys=history_frame_keys,
        )
        device_type, device_index = _device_key(q_blocks.device)
        overlap = _cached_frame_block_overlap(
            device_type,
            device_index,
            history_frames,
            frame_seq,
            block_k,
            history_block_count,
        )
        eligible = overlap[frame_ids].any(dim=-2)

        history_keys = k_blocks[:, :history_block_count].permute(0, 2, 1, 3)
        scores = torch.matmul(
            q_blocks.detach().permute(0, 2, 1, 3).float(),
            history_keys.transpose(-1, -2).float(),
        )
        scores.masked_fill_(~eligible, float("-inf"))

        # This lower bound prevents top-k from admitting ineligible blocks when
        # adjacent selected frames share an execution-block boundary.
        minimum_candidates = frame_ids.shape[-1] * max(1, frame_seq // block_k)
        keep = min(history_keep_blocks, history_block_count, minimum_candidates)
        return torch.topk(scores, keep, dim=-1, sorted=False).indices


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


def _portable_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    block_q: int,
    block_k: int,
    query_block_batch: int,
    scale: float | None,
) -> torch.Tensor:
    """Reference backend that materializes selected K/V blocks."""
    b, _, heads, dim = q.shape
    q_count = q.shape[1] // block_q
    k_count = k.shape[1] // block_k
    k_blocks = k.reshape(b, k_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    v_blocks = v.reshape(b, k_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    q_heads = q.permute(0, 2, 1, 3)
    outputs = []
    for start in range(0, q_count, query_block_batch):
        end = min(start + query_block_batch, q_count)
        selected = block_lut[:, :, start:end]
        selected_k = _gather_query_blocks(k_blocks, selected).flatten(3, 4)
        selected_v = _gather_query_blocks(v_blocks, selected).flatten(3, 4)
        q_group = q_heads[:, :, start * block_q : end * block_q].reshape(
            b, heads, end - start, block_q, dim
        )
        batch_heads_queries = b * heads * (end - start)
        out = F.scaled_dot_product_attention(
            q_group.reshape(batch_heads_queries, block_q, dim),
            selected_k.reshape(batch_heads_queries, selected_k.shape[-2], dim),
            selected_v.reshape(batch_heads_queries, selected_v.shape[-2], dim),
            scale=scale,
        )
        outputs.append(out.reshape(b, heads, (end - start) * block_q, dim))
    result = torch.cat(outputs, dim=2)[:, :, : q.shape[1]]
    return result.permute(0, 2, 1, 3).contiguous()


def _run_sparse_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    config: SparseAttentionConfig,
) -> torch.Tensor:
    backend = config.backend
    if backend in {"auto", "mindiesd"}:
        from .sparse_attention_mindiesd import (
            mindiesd_available,
            mindiesd_sparse_attention_blhd,
            mindiesd_unavailable_reason,
        )

        use_mindiesd = (
            q.device.type == "npu"
            and not torch.is_grad_enabled()
            and config.block_q == 128
            and config.block_k == 128
            and mindiesd_available()
        )
        if use_mindiesd:
            return mindiesd_sparse_attention_blhd(
                q,
                k,
                v,
                block_lut,
                scale=config.softmax_scale,
            )
        if backend == "mindiesd":
            if q.device.type != "npu":
                reason = f"q is on {q.device.type}, not npu"
            elif torch.is_grad_enabled():
                reason = "MindIE-SD RainFusionAttention is forward-only"
            elif config.block_q != 128 or config.block_k != 128:
                reason = "MindIE-SD RainFusionAttention requires block_q=block_k=128"
            else:
                reason = mindiesd_unavailable_reason()
            raise RuntimeError(f"MindIE-SD HSA was requested but is unavailable: {reason}")

    if backend in {"auto", "ascend_triton"}:
        from .sparse_attention_ascend import (
            ascend_triton_available,
            ascend_triton_sparse_attention,
            ascend_triton_sparse_attention_blhd,
            ascend_triton_unavailable_reason,
        )

        use_ascend = q.device.type == "npu" and (
            backend == "ascend_triton" or ascend_triton_available()
        )
        if use_ascend:
            if not torch.is_grad_enabled() and _ASCEND_BLHD_INFERENCE:
                return ascend_triton_sparse_attention_blhd(
                    q,
                    k,
                    v,
                    block_lut,
                    block_q=config.block_q,
                    block_k=config.block_k,
                    scale=config.softmax_scale,
                    validate_lut=_ASCEND_VALIDATE_LUT,
                )
            return ascend_triton_sparse_attention(
                q.permute(0, 2, 1, 3).contiguous(),
                k.permute(0, 2, 1, 3).contiguous(),
                v.permute(0, 2, 1, 3).contiguous(),
                block_lut,
                block_q=config.block_q,
                block_k=config.block_k,
                scale=config.softmax_scale,
                validate_lut=_ASCEND_VALIDATE_LUT,
            ).permute(0, 2, 1, 3).contiguous()
        if backend == "ascend_triton":
            reason = ascend_triton_unavailable_reason()
            if q.device.type != "npu":
                reason = f"q is on {q.device.type}, not npu"
            raise RuntimeError(f"Ascend Triton HSA was requested but is unavailable: {reason}")

    return _portable_sparse_attention(
        q,
        k,
        v,
        block_lut,
        block_q=config.block_q,
        block_k=config.block_k,
        query_block_batch=config.query_block_batch,
        scale=config.softmax_scale,
    )


def hierarchical_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | SparseAttentionConfig | None,
    routing_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply HSA to cached rectangular BLHD tensors using the configured backend."""
    config = (
        sparse_config
        if isinstance(sparse_config, SparseAttentionConfig)
        else SparseAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled:
        return _dense_attention(q, k, v, config.softmax_scale)
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("HSA expects q/k/v in BLHD layout with compatible shapes.")
    if frame_seq <= 0 or config.block_k > frame_seq:
        raise ValueError(
            f"frame_seq ({frame_seq}) must be positive and at least block_k "
            f"({config.block_k})."
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
    if q_work.shape[1] % block_q or k_work.shape[1] % block_k:
        raise ValueError(
            "q and k token lengths must be divisible by block_q/block_k; "
            f"got {q_work.shape[1]}/{k_work.shape[1]} and {block_q}/{block_k}."
        )
    q_count = q_work.shape[1] // block_q
    k_count = k_work.shape[1] // block_k
    history_block_count = history_tokens // block_k
    device_type, device_index = _device_key(q.device)
    current_block_ids = _cached_arange(
        device_type, device_index, history_block_count, k_count
    )
    history_keep = max(1, math.ceil((1.0 - sparsity) * history_block_count))

    q_blocks = q_work.reshape(b, q_count, block_q, heads, dim).mean(dim=2)

    # Historical K is unchanged across the denoising steps of one AR chunk.
    # Cache only its routing summaries; query summaries remain step-dependent.
    cache_key = (
        int(chunk_id), history_tokens, history_frames, frame_seq, block_k,
        heads, dim, q_work.dtype, q_work.device.type, q_work.device.index,
    )
    cached = None
    if routing_cache is not None and not torch.is_grad_enabled():
        if routing_cache.get("key") == cache_key:
            cached = routing_cache
    if cached is None:
        history_k = k_work[:, :history_tokens]
        history_block_means = history_k.reshape(
            b, history_block_count, block_k, heads, dim
        ).mean(dim=2).float()
        history_frame_keys = history_k.reshape(
            b, history_frames, frame_seq, heads, dim
        ).mean(dim=2).float()
        if routing_cache is not None and not torch.is_grad_enabled():
            routing_cache.clear()
            routing_cache.update({
                "key": cache_key,
                "block_means": history_block_means,
                "frame_keys": history_frame_keys,
            })
    else:
        history_block_means = cached["block_means"]
        history_frame_keys = cached["frame_keys"]

    history_ids = _history_block_indices(
        q_blocks,
        history_block_means,
        k_work[:, :history_tokens],
        history_frames,
        frame_seq,
        block_k,
        history_keep,
        config,
        history_frame_keys=history_frame_keys,
    )

    # RainFusion requires ascending indices. Current blocks form a consecutive
    # suffix, so sorting only the shorter historical selection is sufficient.
    selected = torch.sort(history_ids, dim=-1).values
    if config.dense_current:
        current = current_block_ids.view(1, 1, 1, -1).expand(
            b, heads, q_count, -1
        )
        selected = torch.cat([selected, current], dim=-1)
    block_lut = selected.contiguous()
    return _run_sparse_backend(q_work, k_work, v_work, block_lut, config)


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
