import inspect

import pytest
import torch

from wan_5b.modules.hsa_sla_attention import (
    HSASLAAttentionConfig,
    build_hsa_sla_block_lut,
    hsa_sla_cag_attention,
)
from wan_5b.modules.sla_attention import _portable_sparse_attention
from wan_5b.modules.sparse_attention import (
    calculate_chunk_sparsities,
    clear_sparse_attention_cache,
    parse_sparse_config,
    sparse_method,
)
from wan_5b.modules.sparse_routing import (
    SparseKVLayout,
    candidate_block_mask,
    select_hsa_history_frames,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "method": "hsa_sla_cag",
        "backend": "portable",
        "sparsity": 0.5,
        "sparsity_base": 0.75,
        "block_q": 2,
        "block_k": 2,
        "feature_map": "softmax",
        "protect_current_frames": True,
        "protect_longlive_sink_frames": True,
        "keep_near_history_frames": 1,
        "keep_dynamic_history_frames": 1,
        "dense_current_blocks": False,
        "max_global_sink_frames": 2,
        "max_shot_sink_frames": 2,
        "first_chunk_dense": True,
        "linear_cache": True,
    }
    values.update(overrides)
    return HSASLAAttentionConfig.from_mapping(values)


def _layout(history=4, current=2, global_sink=(0,), shot_sink=()):
    return SparseKVLayout(
        resident_frames=history + current,
        history_frames=history,
        current_frames=current,
        global_sink_frames=global_sink,
        shot_sink_frames=shot_sink,
    )


def _projection(dim, identity=False):
    layer = torch.nn.Linear(dim, dim)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dim) if identity else torch.zeros(dim, dim))
        layer.bias.zero_()
    return layer


def test_hybrid_dispatcher_uses_independent_method():
    config = {**_config().__dict__, "method": "hsa_sla_cag"}
    assert sparse_method(config) == "hsa_sla_cag"
    parsed = parse_sparse_config(config)
    assert isinstance(parsed, HSASLAAttentionConfig)


def test_removed_hybrid_fields_fail_fast():
    with pytest.raises(ValueError, match="removed HSA-SLA config fields"):
        HSASLAAttentionConfig.from_mapping(
            {
                **_config().__dict__,
                "candidate_frames": 8,
                "hard_keep_sink_frames": 1,
                "hard_keep_recent_frames": 1,
            }
        )


def test_hybrid_calls_shared_history_router():
    source = inspect.getsource(build_hsa_sla_block_lut)
    assert "select_hsa_history_frames" in source
    assert "current_frame_keys" not in source


def test_hybrid_first_chunk_is_exact_dense_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 4)
    actual = hsa_sla_cag_attention(
        q, q, q, frame_seq=2, chunk_id=0,
        sparse_config=_config(), linear_projection=_projection(4, identity=True),
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_hybrid_current_frames_do_not_make_current_blocks_dense():
    q = torch.ones(1, 4, 1, 2)
    k = torch.ones(1, 12, 1, 2)
    k[:, -4:] = -100.0
    selection = select_hsa_history_frames(
        q_frame_repr=torch.ones(1, 2, 1, 2),
        history_frame_repr=torch.ones(1, 4, 1, 2),
        layout=_layout(),
        keep_near_history_frames=1,
        keep_dynamic_history_frames=1,
        max_global_sink_frames=2,
        max_shot_sink_frames=2,
    )
    eligible = candidate_block_mask(
        selected_history_frame_ids=selection.frame_ids,
        layout=_layout(),
        frame_seq=2,
        block_k=2,
        block_count=6,
    )
    assert torch.all(eligible[..., 4:])
    lut = build_hsa_sla_block_lut(
        q, k, frame_seq=2, sparsity=0.5, config=_config(), kv_layout=_layout()
    )
    assert lut.shape[-1] == 3
    current_hits = torch.isin(torch.tensor([4, 5]), lut.flatten())
    assert not current_hits.all()


def test_hybrid_uses_full_resident_denominator():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 16, 2, 4)
    lut = build_hsa_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=0.5,
        config=_config(),
        kv_layout=_layout(history=6, current=2),
    )
    assert lut.shape == (1, 2, 2, 4)


def test_hybrid_candidate_smaller_than_budget_clamps_without_error():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 16, 1, 2)
    config = _config(
        sparsity=0.0,
        keep_near_history_frames=0,
        keep_dynamic_history_frames=0,
        max_global_sink_frames=0,
        max_shot_sink_frames=0,
    )
    lut = build_hsa_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=0.0,
        config=config,
        kv_layout=_layout(history=6, current=2, global_sink=()),
    )
    assert lut.shape[-1] == 2


def test_hybrid_router_limits_sink_candidates_to_two_frames_each():
    layout = _layout(
        history=6,
        current=2,
        global_sink=(0, 1, 2),
        shot_sink=(3, 4, 5),
    )
    selection = select_hsa_history_frames(
        q_frame_repr=torch.randn(1, 2, 1, 2),
        history_frame_repr=torch.randn(1, 6, 1, 2),
        layout=layout,
        keep_near_history_frames=0,
        keep_dynamic_history_frames=0,
        max_global_sink_frames=2,
        max_shot_sink_frames=2,
    )
    assert selection.protected_history_frames == (0, 1, 3, 4)
    assert selection.selected_history_count == 4


def test_hybrid_global_block_ids_are_invariant_to_head_sharding():
    torch.manual_seed(22)
    q = torch.randn(1, 4, 4, 4)
    k = torch.randn(1, 12, 4, 4)
    config = _config()
    layout = _layout()
    full = build_hsa_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=0.5,
        config=config,
        kv_layout=layout,
    )
    shards = [
        build_hsa_sla_block_lut(
            q[:, :, start:start + 2],
            k[:, :, start:start + 2],
            frame_seq=2,
            sparsity=0.5,
            config=config,
            kv_layout=layout,
        )
        for start in (0, 2)
    ]
    torch.testing.assert_close(torch.cat(shards, dim=1), full)


def test_zero_linear_projection_matches_hybrid_sparse_branch():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    config = _config()
    layout = _layout()
    lut = build_hsa_sla_block_lut(
        q, k, frame_seq=2, sparsity=config.sparsity, config=config,
        kv_layout=layout,
    )
    reference = _portable_sparse_attention(
        q, k, v, lut, block_q=2, block_k=2, query_block_batch=1, scale=None
    )
    actual = hsa_sla_cag_attention(
        q, k, v, frame_seq=2, chunk_id=1, sparse_config=config,
        linear_projection=_projection(4), kv_layout=layout,
    )
    torch.testing.assert_close(actual, reference)


def test_hybrid_linear_projection_and_qkv_receive_gradients():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    projection = _projection(4)
    output = hsa_sla_cag_attention(
        q, k, v, frame_seq=2, chunk_id=1, sparse_config=_config(),
        linear_projection=projection, kv_layout=_layout(),
    )
    output.square().mean().backward()
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert all(tensor.grad is not None for tensor in (q, k, v))


def test_hybrid_cache_reuses_history_but_refreshes_current_summaries():
    torch.manual_seed(5)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    cache = {}
    with torch.no_grad():
        first = build_hsa_sla_block_lut(
            q, k, frame_seq=2, sparsity=0.5, config=_config(),
            kv_layout=_layout(), cache=cache, cache_token=1,
        )
        history_blocks = cache["history_block_keys"]
        changed = k.clone()
        changed[:, -4:] *= -10
        cached = build_hsa_sla_block_lut(
            q, changed, frame_seq=2, sparsity=0.5, config=_config(),
            kv_layout=_layout(), cache=cache, cache_token=1,
        )
        uncached = build_hsa_sla_block_lut(
            q, changed, frame_seq=2, sparsity=0.5, config=_config(),
            kv_layout=_layout(),
        )
    assert cache["history_block_keys"] is history_blocks
    torch.testing.assert_close(cached, uncached)
    assert first.shape == cached.shape


def test_hybrid_longlive_schedule_uses_common_85_95_budget():
    config = {**_config(sparsity=0.85, sparsity_base=0.95).__dict__, "method": "hsa_sla_cag"}
    schedule = calculate_chunk_sparsities(128, 8, 32, config)
    assert schedule[0] == 0.0
    assert schedule[1] == pytest.approx(0.754954, abs=1.0e-6)
    assert schedule[-1] == pytest.approx(0.881041, abs=1.0e-6)


def test_hybrid_router_has_no_device_to_host_item_sync():
    assert ".item()" not in inspect.getsource(build_hsa_sla_block_lut)


def test_sparse_cache_reset_clears_current_and_legacy_keys_only():
    k = torch.ones(1)
    caches = [
        {"k": k, "sparse_attention_cache": {"router": object()}, "sla_attention_cache": {}},
        {"k": k, "sparse_attention_cache": {"router": object()}},
    ]
    clear_sparse_attention_cache(caches)
    assert all("sparse_attention_cache" not in cache for cache in caches)
    assert all("sla_attention_cache" not in cache for cache in caches)
    assert all(cache["k"] is k for cache in caches)
