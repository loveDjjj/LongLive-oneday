# Copyright 2026 LongLive HSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical Sparse Attention routing for causal LongLive inference.

HSA first selects important historical latent frames and then ranks execution
blocks covered by those frames. The current autoregressive chunk remains dense,
matching the original HSA+CAG branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import torch

from .sla_attention import _dense_attention, _run_sparse_backend


@lru_cache(maxsize=128)
def _cached_arange(
    device_type: str, device_index: int | None, start: int, end: int
) -> torch.Tensor:
    return torch.arange(start, end, device=torch.device(device_type, device_index))


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
class HSAAttentionConfig:
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
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "HSAAttentionConfig":
        if not value:
            return cls()
        aliases = {"BLKQ": "block_q", "BLKK": "block_k"}
        normalized = {}
        for key, item in dict(value).items():
            key = aliases.get(key, key)
            if key in cls.__dataclass_fields__:
                normalized[key] = item
        if "sparsity_list" in normalized:
            normalized["sparsity_list"] = tuple(
                float(item) for item in normalized["sparsity_list"]
            )
        if "backend" in normalized:
            aliases = {
                "torch": "portable",
                "triton": "ascend_triton",
                "npu_triton": "ascend_triton",
                "mindie_sd": "mindiesd",
                "rainfusion": "mindiesd",
            }
            backend = str(normalized["backend"]).lower()
            normalized["backend"] = aliases.get(backend, backend)
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.backend not in {"portable", "ascend_triton", "mindiesd", "auto"}:
            raise ValueError(f"unsupported HSA backend: {self.backend}")
        for name in ("sparsity", "sparsity_base"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")
        if self.block_q <= 0 or self.block_k <= 0:
            raise ValueError("block_q and block_k must be positive.")
        if min(self.keep_frames, self.keep_sink, self.keep_near) < 0:
            raise ValueError("HSA frame keep counts must be non-negative.")
        if self.enabled and self.keep_frames == 0:
            raise ValueError("keep_frames must be positive when HSA is enabled.")
        if self.enabled and not self.dense_current:
            raise ValueError("HSA requires dense_current=true.")
        if self.keep_sink + self.keep_near > self.keep_frames:
            raise ValueError("keep_sink + keep_near must not exceed keep_frames.")
        if self.query_block_batch <= 0:
            raise ValueError("query_block_batch must be positive.")
        if self.enabled and self.backend == "mindiesd" and (
            self.block_q != 128 or self.block_k != 128
        ):
            raise ValueError("MindIE-SD sparse execution requires 128-token blocks.")


def _required_history_frames(
    q_blocks: torch.Tensor,
    history_k: torch.Tensor,
    history_frames: int,
    frame_seq: int,
    config: HSAAttentionConfig,
    history_frame_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, query_count, heads, _ = q_blocks.shape
    keep_count = min(config.keep_frames, history_frames)
    device_type, device_index = q_blocks.device.type, q_blocks.device.index
    if keep_count >= history_frames:
        return _cached_arange(device_type, device_index, 0, history_frames).view(
            1, 1, 1, -1
        ).expand(batch, heads, query_count, -1)

    sink_count = min(config.keep_sink, keep_count)
    near_count = min(config.keep_near, keep_count - sink_count)
    middle_keep = keep_count - sink_count - near_count
    selected = []
    if sink_count:
        selected.append(
            _cached_arange(device_type, device_index, 0, sink_count)
            .view(1, 1, 1, -1)
            .expand(batch, heads, query_count, -1)
        )
    if near_count:
        selected.append(
            _cached_arange(
                device_type, device_index, history_frames - near_count, history_frames
            )
            .view(1, 1, 1, -1)
            .expand(batch, heads, query_count, -1)
        )
    if middle_keep:
        frame_keys = history_frame_keys
        if frame_keys is None:
            frame_keys = history_k.reshape(
                batch, history_frames, frame_seq, heads, -1
            ).mean(dim=2)
        middle_keys = frame_keys[:, sink_count : history_frames - near_count]
        scores = torch.matmul(
            q_blocks.permute(0, 2, 1, 3).float(),
            middle_keys.permute(0, 2, 3, 1).float(),
        )
        selected.append(
            torch.topk(scores, middle_keep, dim=-1, sorted=False).indices + sink_count
        )
    return torch.cat(selected, dim=-1)


def _history_block_indices(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    history_k: torch.Tensor,
    history_frames: int,
    frame_seq: int,
    block_k: int,
    history_keep_blocks: int,
    config: HSAAttentionConfig,
    history_frame_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, query_count, heads, _ = q_blocks.shape
    history_block_count = history_k.shape[1] // block_k
    device_type, device_index = q_blocks.device.type, q_blocks.device.index
    if history_keep_blocks >= history_block_count:
        return _cached_arange(
            device_type, device_index, 0, history_block_count
        ).view(1, 1, 1, -1).expand(batch, heads, query_count, -1)

    with torch.no_grad():
        frame_ids = _required_history_frames(
            q_blocks.detach(), history_k.detach(), history_frames, frame_seq,
            config, history_frame_keys,
        )
        overlap = _cached_frame_block_overlap(
            device_type, device_index, history_frames, frame_seq, block_k,
            history_block_count,
        )
        eligible = overlap[frame_ids].any(dim=-2)
        history_keys = k_blocks[:, :history_block_count].permute(0, 2, 1, 3)
        scores = torch.matmul(
            q_blocks.detach().permute(0, 2, 1, 3).float(),
            history_keys.transpose(-1, -2).float(),
        )
        scores.masked_fill_(~eligible, float("-inf"))
        minimum_candidates = frame_ids.shape[-1] * max(1, frame_seq // block_k)
        keep = min(history_keep_blocks, history_block_count, minimum_candidates)
        return torch.topk(scores, keep, dim=-1, sorted=False).indices


def build_hsa_block_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    config: HSAAttentionConfig,
    sparsity: float,
    attention_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    history_tokens = k.shape[1] - q.shape[1]
    history_frames = history_tokens // frame_seq
    batch, _, heads, dim = q.shape
    query_count = q.shape[1] // config.block_q
    key_count = k.shape[1] // config.block_k
    history_block_count = history_tokens // config.block_k
    q_blocks = q.reshape(
        batch, query_count, config.block_q, heads, dim
    ).mean(dim=2)

    cache_key = (
        int(chunk_id), history_tokens, history_frames, frame_seq, config.block_k,
        heads, dim, q.dtype, q.device.type, q.device.index,
    )
    cached = attention_cache if attention_cache and attention_cache.get("key") == cache_key else None
    if cached is None:
        history_k = k[:, :history_tokens]
        block_means = history_k.reshape(
            batch, history_block_count, config.block_k, heads, dim
        ).mean(dim=2).float()
        frame_keys = history_k.reshape(
            batch, history_frames, frame_seq, heads, dim
        ).mean(dim=2).float()
        if attention_cache is not None and not torch.is_grad_enabled():
            attention_cache.clear()
            attention_cache.update(
                {"key": cache_key, "block_means": block_means, "frame_keys": frame_keys}
            )
    else:
        block_means = cached["block_means"]
        frame_keys = cached["frame_keys"]

    history_keep = max(1, math.ceil((1.0 - sparsity) * history_block_count))
    history_ids = _history_block_indices(
        q_blocks, block_means, k[:, :history_tokens], history_frames, frame_seq,
        config.block_k, history_keep, config, frame_keys,
    )
    selected = torch.sort(history_ids, dim=-1).values
    current_ids = _cached_arange(
        q.device.type, q.device.index, history_block_count, key_count
    ).view(1, 1, 1, -1).expand(batch, heads, query_count, -1)
    selected = torch.cat((selected, current_ids), dim=-1).contiguous()
    return selected


def hsa_cag_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | HSAAttentionConfig | None,
    attention_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    config = (
        sparse_config
        if isinstance(sparse_config, HSAAttentionConfig)
        else HSAAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled:
        return _dense_attention(q, k, v, config.softmax_scale)
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("HSA expects q/k/v in compatible BLHD layouts.")
    if frame_seq <= 0 or config.block_k > frame_seq:
        raise ValueError("frame_seq must be positive and at least block_k.")
    if q.shape[1] % frame_seq or k.shape[1] % frame_seq:
        raise ValueError("q and k must contain complete latent frames.")
    if q.shape[1] % config.block_q or k.shape[1] % config.block_k:
        raise ValueError("q/k lengths must be divisible by the configured HSA blocks.")

    history_tokens = k.shape[1] - q.shape[1]
    history_frames = history_tokens // frame_seq
    if history_tokens <= 0 or history_frames < config.min_sparse_history_frames:
        return _dense_attention(q, k, v, config.softmax_scale)
    sparsity = config.sparsity_list[
        min(chunk_id, len(config.sparsity_list) - 1)
    ] if config.sparsity_list else (0.0 if chunk_id <= 0 else config.sparsity)
    if sparsity <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)

    block_lut = build_hsa_block_lut(
        q,
        k,
        frame_seq=frame_seq,
        chunk_id=chunk_id,
        config=config,
        sparsity=sparsity,
        attention_cache=attention_cache,
    )
    return _run_sparse_backend(q, k, v, block_lut, config)


__all__ = [
    "HSAAttentionConfig", "build_hsa_block_lut", "hsa_cag_attention",
    "_history_block_indices",
]
