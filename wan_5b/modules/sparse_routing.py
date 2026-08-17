# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared routing primitives for LongLive sparse attention."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch


_DEBUG_EMITTED: set[tuple[str, int]] = set()


@dataclass(frozen=True)
class SparseKVLayout:
    """Frame metadata for the resident KV tensor passed to attention.

    Frame ids are indices in the assembled resident KV tensor, after global and
    shot sinks have been prepended to the local window.
    """

    resident_frames: int
    history_frames: int
    current_frames: int
    global_sink_frames: tuple[int, ...] = ()
    shot_sink_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if min(self.resident_frames, self.history_frames, self.current_frames) < 0:
            raise ValueError("KV layout frame counts must be non-negative.")
        if self.history_frames + self.current_frames != self.resident_frames:
            raise ValueError(
                "history_frames + current_frames must equal resident_frames."
            )
        for frame_id in self.global_sink_frames + self.shot_sink_frames:
            if not 0 <= frame_id < self.resident_frames:
                raise ValueError(f"protected frame id {frame_id} is outside resident KV.")

    @classmethod
    def contiguous(cls, resident_frames: int, current_frames: int) -> "SparseKVLayout":
        return cls(
            resident_frames=resident_frames,
            history_frames=max(0, resident_frames - current_frames),
            current_frames=current_frames,
        )

    @property
    def protected_history_frames(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    frame_id
                    for frame_id in self.global_sink_frames + self.shot_sink_frames
                    if frame_id < self.history_frames
                }
            )
        )

    def router_protected_history_frames(
        self,
        *,
        max_global_sink_frames: int | None = None,
        max_shot_sink_frames: int | None = None,
    ) -> tuple[int, ...]:
        """Return sink frames allowed to bypass HSA history routing.

        The KV manager may retain an entire sink chunk, while a sparse router
        can protect only a bounded prefix of each sink class. The remaining
        resident sink frames stay available to dynamic scoring and the linear
        compensation branch.
        """
        global_ids = self.global_sink_frames
        shot_ids = self.shot_sink_frames
        if max_global_sink_frames is not None:
            global_ids = global_ids[: max(0, max_global_sink_frames)]
        if max_shot_sink_frames is not None:
            shot_ids = shot_ids[: max(0, max_shot_sink_frames)]
        return tuple(
            sorted(
                {
                    frame_id
                    for frame_id in global_ids + shot_ids
                    if frame_id < self.history_frames
                }
            )
        )

    @property
    def current_frame_ids(self) -> tuple[int, ...]:
        return tuple(range(self.history_frames, self.resident_frames))


@dataclass(frozen=True)
class HSAFrameSelection:
    frame_ids: torch.Tensor
    protected_history_frames: tuple[int, ...]
    near_history_frames: tuple[int, ...]
    dynamic_history_count: int

    @property
    def selected_history_count(self) -> int:
        return self.frame_ids.shape[-1]


def build_sparse_kv_layout(
    *,
    total_tokens: int,
    current_tokens: int,
    frame_seq: int,
    global_sink_token_ranges: tuple[tuple[int, int], ...] = (),
    shot_sink_token_ranges: tuple[tuple[int, int], ...] = (),
) -> SparseKVLayout:
    """Build frame metadata from ranges in the assembled resident KV tensor."""
    if frame_seq <= 0 or total_tokens % frame_seq or current_tokens % frame_seq:
        raise ValueError("KV layout requires complete latent frames.")
    if not 0 <= current_tokens <= total_tokens:
        raise ValueError("current_tokens must be within resident KV.")

    def frame_ids(ranges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
        output: set[int] = set()
        for start, end in ranges:
            if start % frame_seq or end % frame_seq:
                raise ValueError("protected KV token ranges must be frame aligned.")
            if not 0 <= start <= end <= total_tokens:
                raise ValueError("protected KV token range is outside resident KV.")
            output.update(range(start // frame_seq, end // frame_seq))
        return tuple(sorted(output))

    resident_frames = total_tokens // frame_seq
    current_frames = current_tokens // frame_seq
    return SparseKVLayout(
        resident_frames=resident_frames,
        history_frames=resident_frames - current_frames,
        current_frames=current_frames,
        global_sink_frames=frame_ids(global_sink_token_ranges),
        shot_sink_frames=frame_ids(shot_sink_token_ranges),
    )


@lru_cache(maxsize=128)
def cached_arange(
    device_type: str, device_index: int | None, start: int, end: int
) -> torch.Tensor:
    return torch.arange(start, end, device=torch.device(device_type, device_index))


@lru_cache(maxsize=128)
def cached_frame_block_overlap(
    device_type: str,
    device_index: int | None,
    frame_count: int,
    frame_seq: int,
    block_k: int,
    block_count: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    frame_starts = torch.arange(frame_count, device=device) * frame_seq
    frame_ends = frame_starts + frame_seq
    block_starts = torch.arange(block_count, device=device) * block_k
    block_ends = block_starts + block_k
    return (
        (block_starts.unsqueeze(0) < frame_ends.unsqueeze(1))
        & (block_ends.unsqueeze(0) > frame_starts.unsqueeze(1))
    )


def resolve_global_block_budget(
    *, sparsity: float, full_block_count: int, candidate_block_count: int
) -> int:
    """Resolve CAG K against full resident KV, then clamp to candidates."""
    if full_block_count <= 0 or candidate_block_count <= 0:
        return 0
    requested = max(
        1,
        math.floor((1.0 - sparsity) * full_block_count + 1.0e-9),
    )
    return min(requested, full_block_count, candidate_block_count)


def select_hsa_history_frames(
    *,
    q_frame_repr: torch.Tensor,
    history_frame_repr: torch.Tensor,
    layout: SparseKVLayout,
    keep_near_history_frames: int,
    keep_dynamic_history_frames: int,
    max_global_sink_frames: int | None = None,
    max_shot_sink_frames: int | None = None,
) -> HSAFrameSelection:
    """Select protected, near, and dynamic frames from history only."""
    if q_frame_repr.ndim != 4 or history_frame_repr.ndim != 4:
        raise ValueError("HSA frame representations must use BQHD and BFHD layouts.")
    if history_frame_repr.shape[1] != layout.history_frames:
        raise ValueError("history frame representation does not match KV layout.")
    if min(keep_near_history_frames, keep_dynamic_history_frames) < 0:
        raise ValueError("HSA history frame keep counts must be non-negative.")

    batch, query_count, heads, _ = q_frame_repr.shape
    device_type = q_frame_repr.device.type
    device_index = q_frame_repr.device.index
    history_frames = layout.history_frames
    protected = layout.router_protected_history_frames(
        max_global_sink_frames=max_global_sink_frames,
        max_shot_sink_frames=max_shot_sink_frames,
    )
    protected_set = set(protected)

    near_candidates = [
        frame_id
        for frame_id in range(history_frames - 1, -1, -1)
        if frame_id not in protected_set
    ]
    near = tuple(sorted(near_candidates[:keep_near_history_frames]))
    fixed_set = protected_set | set(near)
    remaining = tuple(
        frame_id for frame_id in range(history_frames) if frame_id not in fixed_set
    )
    dynamic_count = min(keep_dynamic_history_frames, len(remaining))

    fixed = protected + near
    parts: list[torch.Tensor] = []
    if fixed:
        fixed_ids = torch.tensor(fixed, device=q_frame_repr.device, dtype=torch.long)
        parts.append(
            fixed_ids.view(1, 1, 1, -1).expand(batch, heads, query_count, -1)
        )
    if dynamic_count:
        remaining_ids = torch.tensor(
            remaining, device=q_frame_repr.device, dtype=torch.long
        )
        remaining_keys = history_frame_repr.index_select(1, remaining_ids)
        scores = torch.matmul(
            q_frame_repr.detach().permute(0, 2, 1, 3).float(),
            remaining_keys.detach().permute(0, 2, 3, 1).float(),
        )
        dynamic_ids = torch.topk(
            scores, dynamic_count, dim=-1, sorted=False
        ).indices
        parts.append(remaining_ids[dynamic_ids])

    if parts:
        selected = torch.sort(torch.cat(parts, dim=-1), dim=-1).values
    else:
        selected = cached_arange(device_type, device_index, 0, 0).view(
            1, 1, 1, 0
        ).expand(batch, heads, query_count, 0)
    return HSAFrameSelection(
        frame_ids=selected,
        protected_history_frames=protected,
        near_history_frames=near,
        dynamic_history_count=dynamic_count,
    )


def candidate_block_mask(
    *,
    selected_history_frame_ids: torch.Tensor,
    layout: SparseKVLayout,
    frame_seq: int,
    block_k: int,
    block_count: int,
) -> torch.Tensor:
    """Map selected history plus all current frames to eligible KV blocks."""
    batch, heads, query_count, _ = selected_history_frame_ids.shape
    current_ids = torch.tensor(
        layout.current_frame_ids,
        device=selected_history_frame_ids.device,
        dtype=torch.long,
    ).view(1, 1, 1, -1).expand(batch, heads, query_count, -1)
    frame_ids = torch.cat((selected_history_frame_ids, current_ids), dim=-1)
    overlap = cached_frame_block_overlap(
        selected_history_frame_ids.device.type,
        selected_history_frame_ids.device.index,
        layout.resident_frames,
        frame_seq,
        block_k,
        block_count,
    )
    return overlap[frame_ids].any(dim=-2)


def emit_sparse_routing_debug(
    *,
    method: str,
    chunk_id: int,
    layout: SparseKVLayout,
    frame_seq: int,
    block_k: int,
    sparsity: float,
    block_lut: torch.Tensor,
    eligible_blocks: torch.Tensor,
    selection: HSAFrameSelection | None = None,
) -> None:
    """Print one representative route per method/chunk when debugging is enabled."""
    if os.environ.get("LONGLIVE_SPARSE_ROUTING_DEBUG", "0") != "1":
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except (ImportError, RuntimeError):
        pass
    key = (method, int(chunk_id))
    if key in _DEBUG_EMITTED:
        return
    _DEBUG_EMITTED.add(key)

    route = block_lut[0, 0, 0]
    eligible = eligible_blocks[0, 0, 0]
    overlap = cached_frame_block_overlap(
        block_lut.device.type,
        block_lut.device.index,
        layout.resident_frames,
        frame_seq,
        block_k,
        eligible.shape[-1],
    )
    current_ids = torch.tensor(
        layout.current_frame_ids, device=block_lut.device, dtype=torch.long
    )
    current_mask = (
        overlap[current_ids].any(dim=0)
        if current_ids.numel()
        else torch.zeros_like(eligible)
    )
    selected_current = int(current_mask[route].sum().item())
    current_blocks = int(current_mask.sum().item())
    candidate_blocks = int(eligible.sum().item())
    final_k = int(route.numel())
    selected_history = final_k - selected_current
    candidate_history_frames = (
        selection.selected_history_count if selection is not None else layout.history_frames
    )
    candidate_total_frames = candidate_history_frames + layout.current_frames
    effective_sparsity = 1.0 - final_k / max(eligible.shape[-1], 1)
    current_hit_ratio = selected_current / max(current_blocks, 1)
    near = selection.near_history_frames if selection is not None else ()
    dynamic = selection.dynamic_history_count if selection is not None else 0
    print(
        "[sparse-route] "
        f"attention_method={method} chunk_id={chunk_id} "
        f"resident_frames={layout.resident_frames} history_frames={layout.history_frames} "
        f"current_frames={layout.current_frames} "
        f"protected_global_sink_frames={layout.global_sink_frames} "
        f"protected_shot_sink_frames={layout.shot_sink_frames} "
        f"protected_current_frames={layout.current_frame_ids} "
        f"near_history_frames={near} dynamic_history_frames={dynamic} "
        f"candidate_history_frames={candidate_history_frames} "
        f"candidate_total_frames={candidate_total_frames} "
        f"full_kv_tokens={layout.resident_frames * frame_seq} "
        f"candidate_kv_tokens={min(candidate_blocks * block_k, layout.resident_frames * frame_seq)} "
        f"full_kv_blocks={eligible.shape[-1]} candidate_blocks={candidate_blocks} "
        f"cag_sparsity={sparsity:.6f} cag_final_k={final_k} "
        f"selected_history_blocks={selected_history} "
        f"selected_current_blocks={selected_current} "
        f"effective_global_block_sparsity={effective_sparsity:.6f} "
        f"current_block_hit_ratio={current_hit_ratio:.6f}"
    )


def reset_sparse_routing_debug() -> None:
    """Allow the next video to emit one route record per method/chunk again."""
    _DEBUG_EMITTED.clear()


__all__ = [
    "HSAFrameSelection",
    "SparseKVLayout",
    "build_sparse_kv_layout",
    "cached_arange",
    "cached_frame_block_overlap",
    "candidate_block_mask",
    "emit_sparse_routing_debug",
    "reset_sparse_routing_debug",
    "resolve_global_block_budget",
    "select_hsa_history_frames",
]
