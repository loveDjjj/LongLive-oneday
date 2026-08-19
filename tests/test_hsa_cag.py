import pytest
import torch

from wan_5b.modules.hsa_attention import (
    HSAAttentionConfig,
    build_hsa_block_lut,
    hsa_cag_attention,
)
from wan_5b.modules.sparse_attention import (
    calculate_chunk_sparsities,
    parse_sparse_config,
    sparse_attention,
    sparse_method,
    with_cag_schedule,
)
from wan_5b.modules.sparse_routing import (
    SparseKVLayout,
    candidate_block_mask,
    resolve_global_block_budget,
    select_hsa_history_frames,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "method": "hsa_cag",
        "backend": "portable",
        "sparsity": 0.5,
        "sparsity_base": 0.75,
        "block_q": 2,
        "block_k": 2,
        "protect_current_frames": True,
        "protect_longlive_sink_frames": True,
        "keep_near_history_frames": 1,
        "keep_dynamic_history_frames": 1,
        "dense_current_blocks": False,
        "hsa_history_mode": "rolling",
        "first_chunk_dense": True,
    }
    values.update(overrides)
    return values


def _layout(history=4, current=2, global_sink=(0,), shot_sink=()):
    return SparseKVLayout(
        resident_frames=history + current,
        history_frames=history,
        current_frames=current,
        global_sink_frames=global_sink,
        shot_sink_frames=shot_sink,
    )


def test_dispatcher_requires_explicit_method():
    assert sparse_method(_config()) == "hsa_cag"
    with pytest.raises(ValueError, match="explicit method"):
        sparse_method({"enabled": True})
    parsed = parse_sparse_config(_config())
    assert isinstance(parsed, HSAAttentionConfig)
    assert parse_sparse_config(parsed) is parsed


def test_removed_hsa_fields_fail_fast():
    with pytest.raises(ValueError, match="removed HSA config fields"):
        HSAAttentionConfig.from_mapping(
            {**_config(), "keep_frames": 8, "dense_current": True}
        )


def test_cag_longlive_128_frame_schedule():
    schedule = calculate_chunk_sparsities(
        128, 8, 32, _config(sparsity=0.85, sparsity_base=0.95)
    )
    expected = {0: 0.0, 1: 0.754954, 3: 0.812082, 7: 0.852477, 15: 0.881041}
    assert len(schedule) == 16
    for index, value in expected.items():
        assert schedule[index] == pytest.approx(value, abs=1.0e-6)


def test_full_history_cag_schedule_uses_full_resident_lengths():
    base = _config(sparsity=0.85, sparsity_base=0.95)
    rolling = with_cag_schedule(
        base,
        num_output_frames=64,
        num_frame_per_block=8,
        local_attn_size=32,
    )
    full = with_cag_schedule(
        {**base, "hsa_history_mode": "full"},
        num_output_frames=64,
        num_frame_per_block=8,
        local_attn_size=32,
    )

    assert full["local_attn_size"] == 32
    assert full["sparsity_list"] == calculate_chunk_sparsities(64, 8, -1, full)
    assert full["sparsity_list"] != rolling["sparsity_list"]


def test_hsa_first_chunk_is_exact_dense_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 4)
    actual = hsa_cag_attention(
        q, q, q, frame_seq=4, chunk_id=0, sparse_config=_config()
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_current_frames_are_candidates_but_current_blocks_are_not_dense():
    q_repr = torch.ones(1, 2, 1, 2)
    history_repr = torch.arange(8, dtype=torch.float32).reshape(1, 4, 1, 2)
    layout = _layout()
    selection = select_hsa_history_frames(
        q_frame_repr=q_repr,
        history_frame_repr=history_repr,
        layout=layout,
        keep_near_history_frames=1,
        keep_dynamic_history_frames=1,
    )
    eligible = candidate_block_mask(
        selected_history_frame_ids=selection.frame_ids,
        layout=layout,
        frame_seq=2,
        block_k=2,
        block_count=6,
    )
    assert torch.all(eligible[..., 4:])

    q = torch.ones(1, 4, 1, 2)
    k = torch.ones(1, 12, 1, 2)
    k[:, -4:] = -100.0
    lut = build_hsa_block_lut(
        q,
        k,
        frame_seq=2,
        chunk_id=1,
        config=HSAAttentionConfig.from_mapping(_config(sparsity=0.5)),
        sparsity=0.5,
        kv_layout=layout,
    )
    assert lut.shape[-1] == 3
    assert not torch.all(torch.isin(torch.tensor([4, 5]), lut.flatten()))


def test_hsa_uses_full_resident_denominator():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 16, 1, 2)
    layout = _layout(history=6, current=2)
    config = HSAAttentionConfig.from_mapping(_config(sparsity=0.5))
    lut = build_hsa_block_lut(
        q, k, frame_seq=2, chunk_id=1, config=config, sparsity=0.5,
        kv_layout=layout,
    )
    assert lut.shape[-1] == 4
    assert resolve_global_block_budget(
        sparsity=0.90,
        full_block_count=220,
        candidate_block_count=150,
    ) == 22


def test_hsa_reuses_history_frame_and_block_summaries():
    torch.manual_seed(21)
    q = torch.randn(1, 4, 1, 2)
    history = torch.randn(1, 8, 1, 2)
    current_a = torch.randn(1, 4, 1, 2)
    current_b = torch.randn(1, 4, 1, 2)
    config = HSAAttentionConfig.from_mapping(_config(sparsity=0.5))
    cache = {}

    with torch.no_grad():
        build_hsa_block_lut(
            q,
            torch.cat((history, current_a), dim=1),
            frame_seq=2,
            chunk_id=1,
            config=config,
            sparsity=0.5,
            kv_layout=_layout(),
            attention_cache=cache,
        )
        frame_keys = cache["frame_keys"]
        history_block_keys = cache["history_block_keys"]
        build_hsa_block_lut(
            q,
            torch.cat((history, current_b), dim=1),
            frame_seq=2,
            chunk_id=1,
            config=config,
            sparsity=0.5,
            kv_layout=_layout(),
            attention_cache=cache,
        )

    assert cache["frame_keys"] is frame_keys
    assert cache["history_block_keys"] is history_block_keys


def test_candidate_pool_smaller_than_budget_clamps_safely():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 16, 1, 2)
    config = HSAAttentionConfig.from_mapping(
        _config(
            sparsity=0.0,
            keep_near_history_frames=0,
            keep_dynamic_history_frames=0,
        )
    )
    lut = build_hsa_block_lut(
        q, k, frame_seq=2, chunk_id=1, config=config, sparsity=0.0,
        kv_layout=_layout(history=6, current=2, global_sink=()),
    )
    assert lut.shape[-1] == 2
    assert resolve_global_block_budget(
        sparsity=0.0, full_block_count=8, candidate_block_count=2
    ) == 2


def test_insufficient_history_and_protected_dedup():
    layout = _layout(history=3, current=2, global_sink=(0, 1), shot_sink=(1,))
    selection = select_hsa_history_frames(
        q_frame_repr=torch.randn(1, 2, 1, 2),
        history_frame_repr=torch.randn(1, 3, 1, 2),
        layout=layout,
        keep_near_history_frames=4,
        keep_dynamic_history_frames=4,
    )
    assert selection.protected_history_frames == (0, 1)
    assert torch.equal(selection.frame_ids, torch.tensor([[[[0, 1, 2], [0, 1, 2]]]]))


def test_multishot_protected_frames_are_never_truncated():
    layout = SparseKVLayout(
        resident_frames=32,
        history_frames=24,
        current_frames=8,
        global_sink_frames=tuple(range(8)),
        shot_sink_frames=tuple(range(8, 16)),
    )
    selection = select_hsa_history_frames(
        q_frame_repr=torch.randn(1, 3, 2, 4),
        history_frame_repr=torch.randn(1, 24, 2, 4),
        layout=layout,
        keep_near_history_frames=4,
        keep_dynamic_history_frames=4,
    )
    assert selection.selected_history_count == 24
    assert set(selection.frame_ids.flatten().tolist()) == set(range(24))


def test_full_history_router_can_select_before_rolling_window():
    q = torch.tensor([[[[1.0, 0.0]]]])
    history = torch.zeros(1, 40, 1, 2)
    history[:, 17, :, 0] = 10.0
    history[:, 5, :, 0] = 5.0
    layout = SparseKVLayout(
        resident_frames=48,
        history_frames=40,
        current_frames=8,
        global_sink_frames=(0, 1),
        shot_sink_frames=(8, 9),
    )

    selection = select_hsa_history_frames(
        q_frame_repr=q,
        history_frame_repr=history,
        layout=layout,
        keep_near_history_frames=4,
        keep_dynamic_history_frames=4,
    )

    frames = set(selection.frame_ids.flatten().tolist())
    assert {0, 1, 8, 9}.issubset(frames)
    assert {36, 37, 38, 39}.issubset(frames)
    assert 17 in frames


def test_history_selection_is_invariant_to_head_sharding():
    torch.manual_seed(4)
    q = torch.randn(1, 3, 4, 4)
    history = torch.randn(1, 12, 4, 4)
    layout = _layout(history=12, current=2, global_sink=(0, 1))
    full = select_hsa_history_frames(
        q_frame_repr=q,
        history_frame_repr=history,
        layout=layout,
        keep_near_history_frames=4,
        keep_dynamic_history_frames=4,
    ).frame_ids
    shards = []
    for start in (0, 2):
        shards.append(
            select_hsa_history_frames(
                q_frame_repr=q[:, :, start:start + 2],
                history_frame_repr=history[:, :, start:start + 2],
                layout=layout,
                keep_near_history_frames=4,
                keep_dynamic_history_frames=4,
            ).frame_ids
        )
    torch.testing.assert_close(torch.cat(shards, dim=1), full)


def test_hsa_global_block_ids_are_invariant_to_head_sharding():
    torch.manual_seed(41)
    q = torch.randn(1, 4, 4, 4)
    k = torch.randn(1, 12, 4, 4)
    config = HSAAttentionConfig.from_mapping(_config())
    layout = _layout()
    full = build_hsa_block_lut(
        q,
        k,
        frame_seq=2,
        chunk_id=1,
        config=config,
        sparsity=0.5,
        kv_layout=layout,
    )
    shards = [
        build_hsa_block_lut(
            q[:, :, start:start + 2],
            k[:, :, start:start + 2],
            frame_seq=2,
            chunk_id=1,
            config=config,
            sparsity=0.5,
            kv_layout=layout,
        )
        for start in (0, 2)
    ]
    torch.testing.assert_close(torch.cat(shards, dim=1), full)


def test_hsa_sparse_path_is_differentiable():
    torch.manual_seed(5)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = sparse_attention(
        q, k, v, frame_seq=2, chunk_id=1,
        sparse_config=_config(), linear_projection=torch.nn.Linear(4, 4),
        kv_layout=_layout(),
    )
    output.square().mean().backward()
    assert all(tensor.grad is not None for tensor in (q, k, v))
