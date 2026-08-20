# Copyright 2026 LongLive HSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""History-routed sparse attention for causal LongLive inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .sla_attention import _dense_attention, _run_sparse_backend
from .sparse_routing import (
    SparseKVLayout,
    candidate_block_mask,
    emit_sparse_routing_debug,
    resolve_global_block_budget,
    select_hsa_history_frames,
)


@dataclass(frozen=True)
class HSAAttentionConfig:
    enabled: bool = False
    backend: str = "portable"
    sparsity: float = 0.85
    sparsity_base: float = 0.95
    block_q: int = 128
    block_k: int = 128
    protect_current_frames: bool = True
    protect_longlive_sink_frames: bool = True
    keep_near_history_frames: int = 4
    keep_dynamic_history_frames: int = 4
    dense_current_blocks: bool = False
    hsa_history_mode: str = "rolling"
    first_chunk_dense: bool = True
    dense_prefix_chunks: int = 1
    budget_reference: str = "full_resident_kv"
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
        removed = {
            "keep_frames", "keep_sink", "keep_near", "dense_current",
            "min_sparse_history_frames",
        }
        stale = removed.intersection(value)
        if stale:
            raise ValueError(
                "removed HSA config fields: " + ", ".join(sorted(stale))
            )
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
        if min(
            self.keep_near_history_frames, self.keep_dynamic_history_frames
        ) < 0:
            raise ValueError("HSA frame keep counts must be non-negative.")
        if self.enabled and not self.protect_current_frames:
            raise ValueError("HSA requires protect_current_frames=true.")
        if self.enabled and not self.protect_longlive_sink_frames:
            raise ValueError("HSA requires protect_longlive_sink_frames=true.")
        if self.enabled and self.dense_current_blocks:
            raise ValueError("HSA requires dense_current_blocks=false.")
        if self.hsa_history_mode not in {"rolling", "full"}:
            raise ValueError("HSA hsa_history_mode must be rolling or full.")
        if self.enabled and not self.first_chunk_dense:
            raise ValueError("HSA requires first_chunk_dense=true.")
        if self.dense_prefix_chunks < 1:
            raise ValueError("HSA dense_prefix_chunks must be at least 1.")
        if self.budget_reference != "full_resident_kv":
            raise ValueError("HSA budget_reference must be full_resident_kv.")
        if self.query_block_batch <= 0:
            raise ValueError("query_block_batch must be positive.")
        if self.enabled and self.backend == "mindiesd" and (
            self.block_q != 128 or self.block_k != 128
        ):
            raise ValueError("MindIE-SD sparse execution requires 128-token blocks.")


def build_hsa_block_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    config: HSAAttentionConfig,
    sparsity: float,
    kv_layout: SparseKVLayout | None = None,
    attention_cache: dict[str, Any] | None = None,
    debug_context: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    history_tokens = k.shape[1] - q.shape[1]
    batch, _, heads, dim = q.shape
    query_count = q.shape[1] // config.block_q
    key_count = k.shape[1] // config.block_k
    current_frames = q.shape[1] // frame_seq
    layout = kv_layout or SparseKVLayout.contiguous(
        k.shape[1] // frame_seq, current_frames
    )
    if layout.history_frames != history_tokens // frame_seq:
        raise ValueError("KV layout history does not match HSA tensors.")
    q_blocks = q.reshape(
        batch, query_count, config.block_q, heads, dim
    ).mean(dim=2)

    cache_key = (
        int(chunk_id), history_tokens, layout.history_frames, frame_seq, config.block_k,
        heads, dim, q.dtype, q.device.type, q.device.index,
    )
    cached = (
        attention_cache
        if attention_cache and attention_cache.get("key") == cache_key
        else None
    )
    if cached is None:
        history_k = k[:, :history_tokens]
        frame_keys = history_k.reshape(
            batch, layout.history_frames, frame_seq, heads, dim
        ).mean(dim=2).float()
        history_block_count = history_tokens // config.block_k
        history_block_keys = history_k.reshape(
            batch, history_block_count, config.block_k, heads, dim
        ).mean(dim=2).permute(0, 2, 1, 3)
        if attention_cache is not None and not torch.is_grad_enabled():
            attention_cache.clear()
            attention_cache.update(
                {
                    "key": cache_key,
                    "frame_keys": frame_keys,
                    "history_block_keys": history_block_keys,
                }
            )
    else:
        frame_keys = cached["frame_keys"]
        history_block_keys = cached["history_block_keys"]

    selection = select_hsa_history_frames(
        q_frame_repr=q_blocks,
        history_frame_repr=frame_keys,
        layout=layout,
        keep_near_history_frames=config.keep_near_history_frames,
        keep_dynamic_history_frames=config.keep_dynamic_history_frames,
    )
    eligible = candidate_block_mask(
        selected_history_frame_ids=selection.frame_ids,
        layout=layout,
        frame_seq=frame_seq,
        block_k=config.block_k,
        block_count=key_count,
    )
    candidate_capacity = min(
        key_count,
        (
            (selection.selected_history_count + layout.current_frames) * frame_seq
            + config.block_k
            - 1
        )
        // config.block_k,
    )
    selected_count = resolve_global_block_budget(
        sparsity=sparsity,
        full_block_count=key_count,
        candidate_block_count=candidate_capacity,
    )
    if selected_count <= 0:
        raise ValueError("HSA candidate pool contains no KV blocks.")

    current_k = k[:, history_tokens:]
    current_block_count = current_k.shape[1] // config.block_k
    current_block_keys = current_k.reshape(
        batch, current_block_count, config.block_k, heads, dim
    ).mean(dim=2).permute(0, 2, 1, 3)
    block_keys = torch.cat((history_block_keys, current_block_keys), dim=2)
    with torch.no_grad():
        scores = torch.matmul(
            q_blocks.detach().permute(0, 2, 1, 3).float(),
            block_keys.detach().transpose(-1, -2).float(),
        )
        scores.masked_fill_(~eligible, float("-inf"))
        selected = torch.topk(
            scores, selected_count, dim=-1, sorted=False
        ).indices
        selected = torch.sort(selected, dim=-1).values
    emit_sparse_routing_debug(
        method="hsa_cag",
        chunk_id=chunk_id,
        layout=layout,
        frame_seq=frame_seq,
        block_k=config.block_k,
        sparsity=sparsity,
        block_lut=selected,
        eligible_blocks=eligible,
        selection=selection,
        debug_context=debug_context,
    )
    return selected.contiguous()


def hsa_cag_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | HSAAttentionConfig | None,
    kv_layout: SparseKVLayout | None = None,
    attention_cache: dict[str, Any] | None = None,
    debug_context: Mapping[str, Any] | None = None,
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
    if history_tokens <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)
    sparsity = config.sparsity_list[
        min(chunk_id, len(config.sparsity_list) - 1)
    ] if config.sparsity_list else (
        0.0 if chunk_id < config.dense_prefix_chunks else config.sparsity
    )
    if sparsity <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)

    block_lut = build_hsa_block_lut(
        q,
        k,
        frame_seq=frame_seq,
        chunk_id=chunk_id,
        config=config,
        sparsity=sparsity,
        kv_layout=kv_layout,
        attention_cache=attention_cache,
        debug_context=debug_context,
    )
    return _run_sparse_backend(q, k, v, block_lut, config)


__all__ = [
    "HSAAttentionConfig", "build_hsa_block_lut", "hsa_cag_attention",
]
