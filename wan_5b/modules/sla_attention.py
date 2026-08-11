# Copyright 2026 LongLive SLA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-Linear Attention with Chunk-Aware Growth for causal video DiTs.

The sparse branch follows SLA's representative-block Top-K routing. CAG
allocates a different total KV budget to each autoregressive chunk, while
configured sink/recent frames remain dense. A trainable
linear-attention projection compensates for information omitted by the sparse
softmax branch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_ASCEND_BLHD_INFERENCE = os.environ.get("LONGLIVE_SLA_BLHD_INFERENCE", "1") == "1"
_ASCEND_VALIDATE_LUT = os.environ.get("LONGLIVE_SLA_VALIDATE_LUT", "0") == "1"


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


@lru_cache(maxsize=128)
def _cached_arange(
    device_type: str,
    device_index: int | None,
    start: int,
    end: int,
) -> torch.Tensor:
    return torch.arange(start, end, device=torch.device(device_type, device_index))


@dataclass(frozen=True)
class SLAAttentionConfig:
    enabled: bool = False
    backend: str = "portable"
    sparsity: float = 0.95
    sparsity_base: float = 0.97
    block_q: int = 64
    block_k: int = 64
    feature_map: str = "softmax"
    keep_sink_frames: int = 1
    keep_recent_frames: int = 1
    dense_current: bool = False
    min_sparse_history_frames: int = 1
    query_block_batch: int = 1
    linear_cache: bool = True
    linear_eps: float = 1.0e-5
    softmax_scale: float | None = None
    num_output_frames: int | None = None
    num_frame_per_block: int = 1
    local_attn_size: int = -1
    sparsity_list: tuple[float, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SLAAttentionConfig":
        if not value:
            return cls()
        aliases = {"BLKQ": "block_q", "BLKK": "block_k", "topk": "topk"}
        normalized: dict[str, Any] = {}
        raw = dict(value)
        for key, item in raw.items():
            key = aliases.get(key, key)
            if key == "topk":
                normalized["sparsity"] = 1.0 - float(item)
            elif key in cls.__dataclass_fields__:
                normalized[key] = item
        if "sparsity_list" in normalized:
            normalized["sparsity_list"] = tuple(
                float(item) for item in normalized["sparsity_list"]
            )
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
        if min(self.keep_sink_frames, self.keep_recent_frames) < 0:
            raise ValueError("SLA frame keep counts must be non-negative.")
        if self.feature_map not in {"softmax", "elu", "relu"}:
            raise ValueError("feature_map must be softmax, elu, or relu.")
        if self.query_block_batch <= 0:
            raise ValueError("query_block_batch must be positive.")
        if self.linear_eps <= 0:
            raise ValueError("linear_eps must be positive.")
        if self.enabled and self.backend == "mindiesd" and (
            self.block_q != 128 or self.block_k != 128
        ):
            raise ValueError("MindIE-SD sparse execution requires 128-token blocks.")


def calculate_chunk_sparsities(
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sparse_config: Mapping[str, Any] | SLAAttentionConfig | None,
) -> list[float]:
    """Allocate CAG sparsity while preserving the requested average FLOPs."""
    config = (
        sparse_config
        if isinstance(sparse_config, SLAAttentionConfig)
        else SLAAttentionConfig.from_mapping(sparse_config)
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
    beta = 0.0 if weighted_flops == 0 else (target_flops - base_flops) / weighted_flops
    tail = [config.sparsity_base - alpha * beta for alpha in alphas]
    return [0.0] + [min(0.999, max(0.0, value)) for value in tail]


def resolve_chunk_sparsity(config: SLAAttentionConfig, chunk_id: int) -> float:
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


def _dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()


def _gather_query_blocks(blocks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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
    """Differentiable block-sparse reference for rectangular Q/KV."""
    batch, _, heads, dim = q.shape
    query_count = q.shape[1] // block_q
    key_count = k.shape[1] // block_k
    k_blocks = k.reshape(batch, key_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    v_blocks = v.reshape(batch, key_count, block_k, heads, dim).permute(0, 3, 1, 2, 4)
    q_heads = q.permute(0, 2, 1, 3)
    outputs = []
    for start in range(0, query_count, query_block_batch):
        end = min(start + query_block_batch, query_count)
        selected = block_lut[:, :, start:end]
        selected_k = _gather_query_blocks(k_blocks, selected).flatten(3, 4)
        selected_v = _gather_query_blocks(v_blocks, selected).flatten(3, 4)
        q_group = q_heads[:, :, start * block_q : end * block_q].reshape(
            batch, heads, end - start, block_q, dim
        )
        merged = batch * heads * (end - start)
        out = F.scaled_dot_product_attention(
            q_group.reshape(merged, block_q, dim),
            selected_k.reshape(merged, selected_k.shape[-2], dim),
            selected_v.reshape(merged, selected_v.shape[-2], dim),
            scale=scale,
        )
        outputs.append(out.reshape(batch, heads, (end - start) * block_q, dim))
    result = torch.cat(outputs, dim=2)[:, :, : q.shape[1]]
    return result.permute(0, 2, 1, 3).contiguous()


def _run_sparse_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    config: SLAAttentionConfig,
) -> torch.Tensor:
    backend = config.backend
    if backend in {"auto", "mindiesd"}:
        from .sla_attention_mindiesd import (
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
                q, k, v, block_lut, scale=config.softmax_scale
            )
        if backend == "mindiesd":
            if q.device.type != "npu":
                reason = f"q is on {q.device.type}, not npu"
            elif torch.is_grad_enabled():
                reason = "MindIE-SD sparse execution is forward-only"
            elif config.block_q != 128 or config.block_k != 128:
                reason = "MindIE-SD sparse execution requires 128-token blocks"
            else:
                reason = mindiesd_unavailable_reason()
            raise RuntimeError(f"MindIE-SD SLA sparse branch is unavailable: {reason}")

    if backend in {"auto", "ascend_triton"}:
        from .sla_attention_ascend import (
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
            reason = (
                f"q is on {q.device.type}, not npu"
                if q.device.type != "npu"
                else ascend_triton_unavailable_reason()
            )
            raise RuntimeError(f"Ascend Triton SLA sparse branch is unavailable: {reason}")

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


def _fixed_key_blocks(
    *,
    key_tokens: int,
    key_blocks: int,
    frame_seq: int,
    block_k: int,
    keep_sink_frames: int,
    keep_recent_frames: int,
    device: torch.device,
) -> torch.Tensor:
    fixed = torch.zeros(key_blocks, dtype=torch.bool, device=device)
    sink_tokens = min(key_tokens, keep_sink_frames * frame_seq)
    if sink_tokens:
        fixed[: math.ceil(sink_tokens / block_k)] = True
    recent_tokens = min(key_tokens, keep_recent_frames * frame_seq)
    if recent_tokens:
        fixed[(key_tokens - recent_tokens) // block_k :] = True
    return fixed


def build_sla_block_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_seq: int,
    sparsity: float,
    config: SLAAttentionConfig,
    cache: dict[str, Any] | None = None,
    cache_token: Any = None,
) -> torch.Tensor:
    """Build a global SLA Top-K LUT with optional fixed sink/recent blocks."""
    batch, query_tokens, heads, dim = q.shape
    history_tokens = k.shape[1] - query_tokens
    query_count = query_tokens // config.block_q
    history_count = history_tokens // config.block_k
    key_count = k.shape[1] // config.block_k
    device_type, device_index = _device_key(q.device)

    q_blocks = q.reshape(
        batch, query_count, config.block_q, heads, dim
    ).mean(dim=2).permute(0, 2, 1, 3)
    cache_key = (
        cache_token,
        history_tokens,
        config.block_k,
        heads,
        dim,
        k.dtype,
        k.device.type,
        k.device.index,
    )
    history_block_keys = None
    if history_count and cache is not None and cache.get("key") == cache_key:
        history_block_keys = cache.get("history_block_keys")
    if history_count and history_block_keys is None:
        history_k = k[:, :history_tokens]
        history_block_keys = history_k.reshape(
            batch, history_count, config.block_k, heads, dim
        ).mean(dim=2).permute(0, 2, 1, 3).float()
        if cache is not None and not torch.is_grad_enabled():
            cache.clear()
            cache.update({"key": cache_key, "history_block_keys": history_block_keys})

    current_count = key_count - history_count
    current_block_keys = k[:, history_tokens:].reshape(
        batch, current_count, config.block_k, heads, dim
    ).mean(dim=2).permute(0, 2, 1, 3).float()
    key_block_keys = (
        current_block_keys
        if history_block_keys is None
        else torch.cat((history_block_keys, current_block_keys), dim=2)
    )

    fixed_mask = _fixed_key_blocks(
        key_tokens=k.shape[1],
        key_blocks=key_count,
        frame_seq=frame_seq,
        block_k=config.block_k,
        keep_sink_frames=config.keep_sink_frames,
        keep_recent_frames=config.keep_recent_frames,
        device=q.device,
    )
    if config.dense_current:
        fixed_mask[history_count:] = True
    selected_count = max(1, math.ceil((1.0 - sparsity) * key_count))
    fixed_ids = torch.nonzero(fixed_mask, as_tuple=False).flatten()
    selected_count = min(key_count, max(selected_count, fixed_ids.numel()))

    with torch.no_grad():
        if selected_count == key_count:
            selected = _cached_arange(
                device_type, device_index, 0, key_count
            ).view(1, 1, 1, -1).expand(batch, heads, query_count, -1)
        else:
            scores = torch.matmul(
                q_blocks.detach().float(), key_block_keys.transpose(-1, -2)
            )
            dynamic_count = selected_count - fixed_ids.numel()
            parts = []
            if fixed_ids.numel():
                parts.append(
                    fixed_ids.view(1, 1, 1, -1).expand(
                        batch, heads, query_count, -1
                    )
                )
                scores[..., fixed_ids] = float("-inf")
            if dynamic_count:
                parts.append(
                    torch.topk(scores, dynamic_count, dim=-1, sorted=False).indices
                )
            selected = torch.cat(parts, dim=-1)
        selected = torch.sort(selected, dim=-1).values

    return selected.contiguous()


def _feature_map(value: torch.Tensor, name: str) -> torch.Tensor:
    if name == "softmax":
        return F.softmax(value, dim=-1)
    if name == "elu":
        return F.elu(value) + 1.0
    if name == "relu":
        return F.relu(value)
    raise ValueError(f"unsupported SLA feature map: {name}")


def _linear_statistics(
    key_features: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    key_heads = key_features.permute(0, 2, 3, 1)
    value_heads = values.permute(0, 2, 1, 3)
    return torch.matmul(key_heads, value_heads), key_features.sum(dim=1)


def _linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    history_tokens: int,
    chunk_id: int,
    config: SLAAttentionConfig,
    cache: dict[str, Any] | None,
) -> torch.Tensor:
    q_features = _feature_map(q, config.feature_map)
    use_cache = (
        config.linear_cache
        and cache is not None
        and not torch.is_grad_enabled()
        and history_tokens > 0
    )
    history_kv = None
    history_sum = None
    cache_key = (
        int(chunk_id),
        history_tokens,
        q.shape[2],
        q.shape[3],
        q.dtype,
        q.device.type,
        q.device.index,
        config.feature_map,
    )
    if use_cache and cache.get("key") == cache_key:
        history_kv = cache.get("kv")
        history_sum = cache.get("sum")
    if history_kv is None and history_tokens:
        history_features = _feature_map(k[:, :history_tokens], config.feature_map)
        history_kv, history_sum = _linear_statistics(
            history_features, v[:, :history_tokens]
        )
        if use_cache:
            cache.clear()
            cache.update({"key": cache_key, "kv": history_kv, "sum": history_sum})

    current_features = _feature_map(k[:, history_tokens:], config.feature_map)
    kv_sum, key_sum = _linear_statistics(
        current_features, v[:, history_tokens:]
    )
    if history_kv is not None:
        kv_sum = kv_sum + history_kv
        key_sum = key_sum + history_sum
    q_heads = q_features.permute(0, 2, 1, 3)
    numerator = torch.matmul(q_heads, kv_sum).permute(0, 2, 1, 3)
    denominator = torch.matmul(
        q_heads, key_sum.unsqueeze(-1)
    ).permute(0, 2, 1, 3)
    return numerator / denominator.clamp_min(config.linear_eps)


def sla_cag_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq: int,
    chunk_id: int,
    sparse_config: Mapping[str, Any] | SLAAttentionConfig | None,
    linear_projection: torch.nn.Module,
    attention_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run rectangular SLA+CAG on BLHD query and rolling cached KV tensors."""
    config = (
        sparse_config
        if isinstance(sparse_config, SLAAttentionConfig)
        else SLAAttentionConfig.from_mapping(sparse_config)
    )
    if not config.enabled:
        return _dense_attention(q, k, v, config.softmax_scale)
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("SLA expects q/k/v in compatible BLHD layouts.")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("SLA requires matching q/k/v head dimensions.")
    if frame_seq <= 0 or config.block_k > frame_seq:
        raise ValueError("frame_seq must be positive and at least block_k.")
    if q.shape[1] % frame_seq or k.shape[1] % frame_seq:
        raise ValueError("q and k must contain complete latent frames.")
    if q.shape[1] % config.block_q or k.shape[1] % config.block_k:
        raise ValueError("q/k lengths must be divisible by the configured SLA blocks.")

    history_tokens = k.shape[1] - q.shape[1]
    history_frames = history_tokens // frame_seq
    sparsity = resolve_chunk_sparsity(config, chunk_id)
    if (
        history_tokens <= 0
        or history_frames < config.min_sparse_history_frames
        or sparsity <= 0
    ):
        return _dense_attention(q, k, v, config.softmax_scale)

    router_cache = None if attention_cache is None else attention_cache.setdefault("router", {})
    linear_cache = None if attention_cache is None else attention_cache.setdefault("linear", {})
    block_lut = build_sla_block_lut(
        q,
        k,
        frame_seq=frame_seq,
        sparsity=sparsity,
        config=config,
        cache=router_cache,
        cache_token=chunk_id,
    )
    sparse_output = _run_sparse_backend(q, k, v, block_lut, config)
    linear_output = _linear_attention(
        q,
        k,
        v,
        history_tokens=history_tokens,
        chunk_id=chunk_id,
        config=config,
        cache=linear_cache,
    )
    projection_dtype = next(linear_projection.parameters()).dtype
    projected = linear_projection(linear_output.to(projection_dtype)).to(q.dtype)
    return sparse_output + projected


def with_cag_schedule(
    sparse_config: Mapping[str, Any] | None,
    *,
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
) -> dict[str, Any]:
    output = dict(sparse_config or {})
    parsed = SLAAttentionConfig.from_mapping(output)
    output["num_output_frames"] = num_output_frames
    output["num_frame_per_block"] = num_frame_per_block
    output["local_attn_size"] = local_attn_size
    output["sparsity_list"] = calculate_chunk_sparsities(
        num_output_frames, num_frame_per_block, local_attn_size, parsed
    )
    return output


__all__ = [
    "SLAAttentionConfig",
    "build_sla_block_lut",
    "calculate_chunk_sparsities",
    "resolve_chunk_sparsity",
    "sla_cag_attention",
    "with_cag_schedule",
]
