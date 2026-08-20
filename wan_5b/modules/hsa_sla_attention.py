# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""Frame-filtered Sparse-Linear Attention with Chunk-Aware Growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .sla_attention import (
    _dense_attention,
    _projected_linear_attention,
    _run_sparse_backend,
)
from .sparse_routing import (
    SparseKVLayout,
    candidate_block_mask,
    emit_sparse_routing_debug,
    resolve_global_block_budget,
    select_hsa_history_frames,
)


@dataclass(frozen=True)
class HSASLAAttentionConfig:
    enabled: bool = False
    backend: str = "portable"
    sparsity: float = 0.85
    sparsity_base: float = 0.95
    block_q: int = 128
    block_k: int = 128
    feature_map: str = "softmax"
    protect_current_frames: bool = True
    protect_longlive_sink_frames: bool = True
    keep_near_history_frames: int = 4
    keep_dynamic_history_frames: int = 4
    dense_current_blocks: bool = False
    max_global_sink_frames: int = 2
    max_shot_sink_frames: int = 2
    first_chunk_dense: bool = True
    dense_prefix_chunks: int = 1
    budget_reference: str = "full_resident_kv"
    full_kv_linear_compensation: bool = True
    query_block_batch: int = 1
    linear_cache: bool = True
    linear_eps: float = 1.0e-5
    softmax_scale: float | None = None
    num_output_frames: int | None = None
    num_frame_per_block: int = 1
    local_attn_size: int = -1
    sparsity_list: tuple[float, ...] = ()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "HSASLAAttentionConfig":
        if not value:
            return cls()
        aliases = {"BLKQ": "block_q", "BLKK": "block_k"}
        removed = {
            "candidate_frames", "keep_frames", "keep_sink_frames",
            "keep_recent_frames", "dense_current", "min_sparse_history_frames",
            "hard_keep_sink_frames", "hard_keep_recent_frames",
        }
        stale = removed.intersection(value)
        if stale:
            raise ValueError(
                "removed HSA-SLA config fields: " + ", ".join(sorted(stale))
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
            raise ValueError(f"unsupported HSA-SLA backend: {self.backend}")
        for name in ("sparsity", "sparsity_base"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")
        if self.block_q <= 0 or self.block_k <= 0:
            raise ValueError("block_q and block_k must be positive.")
        if min(
            self.keep_near_history_frames,
            self.keep_dynamic_history_frames,
            self.max_global_sink_frames,
            self.max_shot_sink_frames,
        ) < 0:
            raise ValueError("fixed frame counts must be non-negative.")
        if self.enabled and not self.protect_current_frames:
            raise ValueError("HSA-SLA requires protect_current_frames=true.")
        if self.enabled and not self.protect_longlive_sink_frames:
            raise ValueError("HSA-SLA requires protect_longlive_sink_frames=true.")
        if self.enabled and self.dense_current_blocks:
            raise ValueError("HSA-SLA requires dense_current_blocks=false.")
        if self.enabled and not self.first_chunk_dense:
            raise ValueError("HSA-SLA requires first_chunk_dense=true.")
        if self.dense_prefix_chunks < 1:
            raise ValueError("HSA-SLA dense_prefix_chunks must be at least 1.")
        if self.budget_reference != "full_resident_kv":
            raise ValueError("HSA-SLA budget_reference must be full_resident_kv.")
        if self.enabled and not self.full_kv_linear_compensation:
            raise ValueError("HSA-SLA requires full_kv_linear_compensation=true.")
        if self.feature_map not in {"softmax", "elu", "relu"}:
            raise ValueError("feature_map must be softmax, elu, or relu.")
        if self.query_block_batch <= 0 or self.linear_eps <= 0:
            raise ValueError("query_block_batch and linear_eps must be positive.")
        if self.enabled and self.backend == "mindiesd" and (
            self.block_q != 128 or self.block_k != 128
        ):
            raise ValueError("MindIE-SD sparse execution requires 128-token blocks.")


def build_hsa_sla_block_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_seq: int,
    sparsity: float,
    config: HSASLAAttentionConfig,
    kv_layout: SparseKVLayout | None = None,
    cache: dict[str, Any] | None = None,
    cache_token: Any = None,
) -> torch.Tensor:
    """Select frames first, then Smooth-K blocks within those candidates."""
    batch, query_tokens, heads, dim = q.shape
    key_tokens = k.shape[1]
    history_tokens = key_tokens - query_tokens
    query_count = query_tokens // config.block_q
    key_count = key_tokens // config.block_k
    history_blocks = history_tokens // config.block_k
    current_frames = query_tokens // frame_seq
    layout = kv_layout or SparseKVLayout.contiguous(
        key_tokens // frame_seq, current_frames
    )
    if layout.history_frames != history_tokens // frame_seq:
        raise ValueError("KV layout history does not match HSA-SLA tensors.")

    q_blocks_blhd = q.reshape(
        batch, query_count, config.block_q, heads, dim
    ).mean(dim=2)
    q_blocks = q_blocks_blhd.permute(0, 2, 1, 3)

    cache_key = (
        cache_token, history_tokens, frame_seq, config.block_k, heads, dim,
        k.dtype, k.device.type, k.device.index,
    )
    cached = cache if cache is not None and cache.get("key") == cache_key else None
    if cached is None:
        history_k = k[:, :history_tokens]
        history_frame_keys = history_k.reshape(
            batch, layout.history_frames, frame_seq, heads, dim
        ).mean(dim=2)
        history_block_keys = history_k.reshape(
            batch, history_blocks, config.block_k, heads, dim
        ).mean(dim=2).permute(0, 2, 1, 3)
        if cache is not None and not torch.is_grad_enabled():
            cache.clear()
            cache.update(
                {
                    "key": cache_key,
                    "history_frame_keys": history_frame_keys,
                    "history_block_keys": history_block_keys,
                }
            )
    else:
        history_frame_keys = cached["history_frame_keys"]
        history_block_keys = cached["history_block_keys"]

    current_block_keys = k[:, history_tokens:].reshape(
        batch, key_count - history_blocks, config.block_k, heads, dim
    ).mean(dim=2).permute(0, 2, 1, 3)
    block_keys = torch.cat((history_block_keys, current_block_keys), dim=2)

    with torch.no_grad():
        selection = select_hsa_history_frames(
            q_frame_repr=q_blocks_blhd,
            history_frame_repr=history_frame_keys,
            layout=layout,
            keep_near_history_frames=config.keep_near_history_frames,
            keep_dynamic_history_frames=config.keep_dynamic_history_frames,
            max_global_sink_frames=config.max_global_sink_frames,
            max_shot_sink_frames=config.max_shot_sink_frames,
        )
        eligible = candidate_block_mask(
            selected_history_frame_ids=selection.frame_ids,
            layout=layout,
            frame_seq=frame_seq,
            block_k=config.block_k,
            block_count=key_count,
        )

        # SLA Smooth-K: center representatives before QK block scoring.
        smooth_keys = block_keys - block_keys.mean(dim=2, keepdim=True)
        scores = torch.matmul(q_blocks.detach(), smooth_keys.transpose(-1, -2))
        scores.masked_fill_(~eligible, float("-inf"))

        candidate_capacity = min(
            key_count,
            (
                (selection.selected_history_count + layout.current_frames) * frame_seq
                + config.block_k
                - 1
            )
            // config.block_k
        )
        selected_count = resolve_global_block_budget(
            sparsity=sparsity,
            full_block_count=key_count,
            candidate_block_count=candidate_capacity,
        )
        selected_count = min(candidate_capacity, selected_count)
        selected = torch.topk(
            scores, selected_count, dim=-1, sorted=False
        ).indices
        selected = torch.sort(selected, dim=-1).values
    emit_sparse_routing_debug(
        method="hsa_sla_cag",
        chunk_id=int(cache_token or 0),
        layout=layout,
        frame_seq=frame_seq,
        block_k=config.block_k,
        sparsity=sparsity,
        block_lut=selected,
        eligible_blocks=eligible,
        selection=selection,
    )
    return selected.contiguous()


def hsa_sla_cag_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | HSASLAAttentionConfig | None,
    linear_projection: torch.nn.Module,
    kv_layout: SparseKVLayout | None = None,
    attention_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    config = (
        sparse_config
        if isinstance(sparse_config, HSASLAAttentionConfig)
        else HSASLAAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled:
        return _dense_attention(q, k, v, config.softmax_scale)
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("HSA-SLA expects q/k/v in compatible BLHD layouts.")
    if frame_seq <= 0 or config.block_k > frame_seq:
        raise ValueError("frame_seq must be positive and at least block_k.")
    if q.shape[1] % frame_seq or k.shape[1] % frame_seq:
        raise ValueError("q and k must contain complete latent frames.")
    if q.shape[1] % config.block_q or k.shape[1] % config.block_k:
        raise ValueError("q/k lengths must be divisible by configured blocks.")

    history_tokens = k.shape[1] - q.shape[1]
    sparsity = config.sparsity_list[
        min(chunk_id, len(config.sparsity_list) - 1)
    ] if config.sparsity_list else (
        0.0 if chunk_id < config.dense_prefix_chunks else config.sparsity
    )
    if history_tokens <= 0 or sparsity <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)

    router_cache = None if attention_cache is None else attention_cache.setdefault("router", {})
    linear_cache = None if attention_cache is None else attention_cache.setdefault("linear", {})
    block_lut = build_hsa_sla_block_lut(
        q, k, frame_seq=frame_seq, sparsity=sparsity, config=config,
        kv_layout=kv_layout,
        cache=router_cache, cache_token=chunk_id,
    )
    sparse_output = _run_sparse_backend(q, k, v, block_lut, config)
    linear_output = _projected_linear_attention(
        q, k, v, history_tokens=history_tokens, chunk_id=chunk_id,
        config=config, cache=linear_cache, projection=linear_projection,
    )
    return sparse_output + linear_output


__all__ = [
    "HSASLAAttentionConfig", "build_hsa_sla_block_lut", "hsa_sla_cag_attention",
]
