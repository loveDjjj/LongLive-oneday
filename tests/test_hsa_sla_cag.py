import torch

from wan_5b.modules.hsa_sla_attention import (
    HSASLAAttentionConfig,
    build_hsa_sla_block_lut,
    hsa_sla_cag_attention,
)
from wan_5b.modules.sla_attention import _portable_sparse_attention
from wan_5b.modules.sparse_attention import parse_sparse_config, sparse_method
from wan_5b.modules.sparse_attention import calculate_chunk_sparsities


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
        "candidate_frames": 3,
        "keep_sink_frames": 1,
        "keep_recent_frames": 1,
        "dense_current": False,
        "min_sparse_history_frames": 1,
        "linear_cache": True,
    }
    values.update(overrides)
    return HSASLAAttentionConfig.from_mapping(values)


def _projection(dim, identity=False):
    layer = torch.nn.Linear(dim, dim)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dim) if identity else torch.zeros(dim, dim))
        layer.bias.zero_()
    return layer


def test_hybrid_dispatcher_uses_independent_method():
    config = {**_config().__dict__, "method": "hsa_sla_cag"}
    assert sparse_method(config) == "hsa_sla_cag"
    assert isinstance(parse_sparse_config(config), HSASLAAttentionConfig)


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


def test_hybrid_lut_keeps_sink_and_recent_with_global_budget():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    config = _config(sparsity=0.5)
    lut = build_hsa_sla_block_lut(
        q, k, frame_seq=2, sparsity=0.5, config=config
    )

    assert lut.shape == (1, 2, 2, 3)
    assert torch.all(lut[..., 0] == 0)
    assert torch.all(lut[..., -1] == 5)
    assert all(row.unique().numel() == 3 for row in lut.flatten(0, 2))


def test_zero_linear_projection_matches_hybrid_sparse_branch():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    config = _config()
    lut = build_hsa_sla_block_lut(
        q, k, frame_seq=2, sparsity=config.sparsity, config=config
    )
    reference = _portable_sparse_attention(
        q, k, v, lut, block_q=2, block_k=2, query_block_batch=1, scale=None
    )
    actual = hsa_sla_cag_attention(
        q, k, v, frame_seq=2, chunk_id=1,
        sparse_config=config, linear_projection=_projection(4),
    )
    torch.testing.assert_close(actual, reference)


def test_hybrid_linear_projection_receives_gradients():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    projection = _projection(4)
    output = hsa_sla_cag_attention(
        q, k, v, frame_seq=2, chunk_id=1,
        sparse_config=_config(), linear_projection=projection,
    )
    output.square().mean().backward()

    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_hybrid_cache_reuses_history_but_refreshes_current_summaries():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    cache = {}
    with torch.no_grad():
        first = build_hsa_sla_block_lut(
            q, k, frame_seq=2, sparsity=0.5, config=_config(),
            cache=cache, cache_token=1,
        )
        history_blocks = cache["history_block_keys"]
        changed = k.clone()
        changed[:, -4:] = changed[:, -4:] * -10
        cached = build_hsa_sla_block_lut(
            q, changed, frame_seq=2, sparsity=0.5, config=_config(),
            cache=cache, cache_token=1,
        )
        uncached = build_hsa_sla_block_lut(
            q, changed, frame_seq=2, sparsity=0.5, config=_config(),
        )

    assert cache["history_block_keys"] is history_blocks
    torch.testing.assert_close(cached, uncached)
    assert first.shape == cached.shape


def test_hybrid_rejects_too_few_candidate_frames_for_budget():
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 16, 1, 2)
    config = _config(
        sparsity=0.25,
        candidate_frames=2,
        keep_sink_frames=1,
        keep_recent_frames=1,
    )
    try:
        build_hsa_sla_block_lut(q, k, frame_seq=2, sparsity=0.25, config=config)
    except ValueError as error:
        assert "candidate_frames" in str(error)
    else:
        raise AssertionError("expected insufficient candidate budget to fail")


def test_hybrid_router_has_no_device_to_host_item_sync():
    import inspect

    source = inspect.getsource(build_hsa_sla_block_lut)
    assert ".item()" not in source


def test_hybrid_cag_budget_is_valid_for_all_release_durations():
    import math

    config = {
        **_config(
            sparsity=0.90,
            sparsity_base=0.93,
            block_q=128,
            block_k=128,
            candidate_frames=8,
        ).__dict__,
        "method": "hsa_sla_cag",
    }
    candidate_capacity = math.ceil(8 * 880 / 128)

    for latent_frames in (32, 192, 384):
        schedule = calculate_chunk_sparsities(
            latent_frames, 8, 32, config
        )
        assert len(schedule) == latent_frames // 8
        assert schedule[0] == 0.0
        for chunk_id, sparsity in enumerate(schedule[1:], start=1):
            resident_frames = min((chunk_id + 1) * 8, 32)
            key_blocks = resident_frames * 880 // 128
            selected = math.ceil((1.0 - sparsity) * key_blocks)
            assert selected <= candidate_capacity
        tail_blocks = 32 * 880 // 128
        tail_selected = math.ceil((1.0 - schedule[-1]) * tail_blocks)
        assert 20 <= tail_selected <= 22
