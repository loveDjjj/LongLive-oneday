# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""Frame-filtered Sparse-Linear Attention with Chunk-Aware Growth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .hsa_attention import _cached_frame_block_overlap, _required_history_frames
from .sla_attention import (
    _cached_arange,
    _dense_attention,
    _fixed_key_blocks,
    _projected_linear_attention,
    _run_sparse_backend,
)


@dataclass(frozen=True)
class HSASLAAttentionConfig:
    enabled: bool = False
    backend: str = "portable"
    sparsity: float = 0.90
    sparsity_base: float = 0.93
    block_q: int = 128
    block_k: int = 128
    feature_map: str = "softmax"
    candidate_frames: int = 8
    keep_sink_frames: int = 1
    keep_recent_frames: int = 1
    dense_current: bool = False
    min_sparse_history_frames: int = 2
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
        aliases = {"BLKQ": "block_q", "BLKK": "block_k", "keep_frames": "candidate_frames"}
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
        if self.candidate_frames <= 0:
            raise ValueError("candidate_frames must be positive.")
        if min(self.keep_sink_frames, self.keep_recent_frames) < 0:
            raise ValueError("fixed frame counts must be non-negative.")
        if self.keep_sink_frames + self.keep_recent_frames > self.candidate_frames:
            raise ValueError(
                "keep_sink_frames + keep_recent_frames must not exceed candidate_frames."
            )
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
    cache: dict[str, Any] | None = None,
    cache_token: Any = None,
) -> torch.Tensor:
    """Select frames first, then Smooth-K blocks within those candidates."""
    batch, query_tokens, heads, dim = q.shape
    key_tokens = k.shape[1]
    history_tokens = key_tokens - query_tokens
    query_count = query_tokens // config.block_q
    key_count = key_tokens // config.block_k
    key_frames = key_tokens // frame_seq
    history_blocks = history_tokens // config.block_k
    history_frames = history_tokens // frame_seq
    device_type, device_index = q.device.type, q.device.index

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
            batch, history_frames, frame_seq, heads, dim
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

    current_k = k[:, history_tokens:]
    current_frames = current_k.shape[1] // frame_seq
    current_blocks = current_k.shape[1] // config.block_k
    current_frame_keys = current_k.reshape(
        batch, current_frames, frame_seq, heads, dim
    ).mean(dim=2)
    current_block_keys = current_k.reshape(
        batch, current_blocks, config.block_k, heads, dim
    ).mean(dim=2).permute(0, 2, 1, 3)
    frame_keys = torch.cat((history_frame_keys, current_frame_keys), dim=1)
    block_keys = torch.cat((history_block_keys, current_block_keys), dim=2)

    # Reuse HSA's fixed sink/near plus dynamic frame selection over resident KV.
    frame_config = type(
        "FrameConfig",
        (),
        {
            "keep_frames": config.candidate_frames,
            "keep_sink": config.keep_sink_frames,
            "keep_near": config.keep_recent_frames,
        },
    )()
    with torch.no_grad():
        frame_ids = _required_history_frames(
            q_blocks_blhd.detach(), k.detach(), key_frames, frame_seq,
            frame_config, frame_keys,
        )
        overlap = _cached_frame_block_overlap(
            device_type, device_index, key_frames, frame_seq, config.block_k,
            key_count,
        )
        eligible = overlap[frame_ids].any(dim=-2)

        # SLA Smooth-K: center representatives before QK block scoring.
        smooth_keys = block_keys - block_keys.mean(dim=2, keepdim=True)
        scores = torch.matmul(q_blocks.detach(), smooth_keys.transpose(-1, -2))
        scores.masked_fill_(~eligible, float("-inf"))

        fixed_mask = _fixed_key_blocks(
            key_tokens=key_tokens,
            key_blocks=key_count,
            frame_seq=frame_seq,
            block_k=config.block_k,
            keep_sink_frames=config.keep_sink_frames,
            keep_recent_frames=config.keep_recent_frames,
            device=q.device,
        )
        if config.dense_current:
            current_count = query_tokens // config.block_k
            fixed_mask[key_count - current_count :] = True
        fixed_ids = torch.nonzero(fixed_mask, as_tuple=False).flatten()
        selected_count = max(1, math.ceil((1.0 - sparsity) * key_count))
        selected_count = min(key_count, max(selected_count, fixed_ids.numel()))
        dynamic_count = selected_count - fixed_ids.numel()
        minimum_candidate_blocks = min(
            key_count,
            math.ceil(min(config.candidate_frames, key_frames) * frame_seq / config.block_k),
        )
        if selected_count > minimum_candidate_blocks:
            raise ValueError(
                f"candidate_frames={config.candidate_frames} cannot cover the CAG "
                f"budget of {selected_count} blocks; increase candidate_frames or sparsity"
            )
        parts = []
        if fixed_ids.numel():
            parts.append(
                fixed_ids.view(1, 1, 1, -1).expand(
                    batch, heads, query_count, -1
                )
            )
            scores[..., fixed_ids] = float("-inf")
        if dynamic_count:
            parts.append(torch.topk(scores, dynamic_count, dim=-1, sorted=False).indices)
        selected = torch.sort(torch.cat(parts, dim=-1), dim=-1).values
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
    history_frames = history_tokens // frame_seq
    sparsity = config.sparsity_list[
        min(chunk_id, len(config.sparsity_list) - 1)
    ] if config.sparsity_list else (0.0 if chunk_id <= 0 else config.sparsity)
    if history_tokens <= 0 or history_frames < config.min_sparse_history_frames or sparsity <= 0:
        return _dense_attention(q, k, v, config.softmax_scale)

    router_cache = None if attention_cache is None else attention_cache.setdefault("router", {})
    linear_cache = None if attention_cache is None else attention_cache.setdefault("linear", {})
    block_lut = build_hsa_sla_block_lut(
        q, k, frame_seq=frame_seq, sparsity=sparsity, config=config,
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
