import torch

from wan_5b.modules.hsa_attention import HSAAttentionConfig, hsa_cag_attention
from wan_5b.modules.sparse_attention import (
    calculate_chunk_sparsities,
    parse_sparse_config,
    sparse_attention,
    sparse_method,
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
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
        "dense_current": True,
        "min_sparse_history_frames": 1,
    }
    values.update(overrides)
    return values


def test_dispatcher_parses_explicit_and_legacy_methods():
    assert sparse_method(_config()) == "hsa_cag"
    assert sparse_method({"enabled": True, "keep_frames": 4}) == "hsa_cag"
    assert sparse_method({"enabled": True, "feature_map": "softmax"}) == "sla_cag"
    parsed = parse_sparse_config(_config())
    assert isinstance(parsed, HSAAttentionConfig)
    assert sparse_method(parsed) == "hsa_cag"
    assert parse_sparse_config(parsed) is parsed


def test_dispatcher_accepts_parsed_hsa_config_in_attention_path():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    output = sparse_attention(
        q,
        k,
        k,
        frame_seq=4,
        chunk_id=2,
        sparse_config=parse_sparse_config(_config()),
        linear_projection=torch.nn.Linear(4, 4),
    )
    assert output.shape == q.shape


def test_hsa_cag_starts_dense_and_preserves_average_budget():
    config = _config(sparsity=0.85, sparsity_base=0.95)
    schedule = calculate_chunk_sparsities(32, 8, 32, config)

    assert len(schedule) == 4
    assert schedule[0] == 0.0
    assert schedule[1] < schedule[-1]
    kv_lengths = [16, 24, 32]
    kept = sum((1.0 - value) * length for value, length in zip(schedule[1:], kv_lengths))
    target = sum((1.0 - config["sparsity"]) * length for length in kv_lengths)
    assert abs(kept - target) < 1.0e-6


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


def test_hsa_keeps_current_chunk_dense_and_is_differentiable():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = sparse_attention(
        q,
        k,
        v,
        frame_seq=4,
        chunk_id=2,
        sparse_config=_config(),
        linear_projection=torch.nn.Linear(4, 4),
    )
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_hsa_routing_cache_reuses_history_summaries():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    cache = {}
    with torch.no_grad():
        first = hsa_cag_attention(
            q, k, v, frame_seq=4, chunk_id=2,
            sparse_config=_config(), attention_cache=cache,
        )
        block_means = cache["block_means"]
        second = hsa_cag_attention(
            q, k, v, frame_seq=4, chunk_id=2,
            sparse_config=_config(), attention_cache=cache,
        )
    assert cache["block_means"] is block_means
    torch.testing.assert_close(second, first)


def test_hsa_rejects_sparse_current_chunk():
    try:
        HSAAttentionConfig.from_mapping(_config(dense_current=False))
    except ValueError as error:
        assert "dense_current=true" in str(error)
    else:
        raise AssertionError("expected HSA sparse-current validation to fail")
