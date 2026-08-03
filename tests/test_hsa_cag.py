import torch

from wan_5b.modules.sparse_attention import (
    SparseAttentionConfig,
    calculate_chunk_sparsities,
    hierarchical_sparse_attention,
)


def test_cag_starts_dense_and_preserves_average_budget():
    config = SparseAttentionConfig(
        enabled=True,
        sparsity=0.85,
        sparsity_base=0.95,
    )
    schedule = calculate_chunk_sparsities(32, 4, 32, config)

    assert schedule[0] == 0.0
    assert len(schedule) == 8
    assert all(0.0 <= value < 1.0 for value in schedule)
    assert schedule[1] < schedule[-1]

    frame_counts = list(range(8, 33, 4))
    kv_lengths = [min(count, 32) for count in frame_counts]
    scheduled = sum((1.0 - value) * length for value, length in zip(schedule[1:], kv_lengths))
    target = sum((1.0 - config.sparsity) * length for length in kv_lengths)
    assert abs(scheduled - target) < 1e-6


def test_hsa_first_chunk_matches_dense_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    v = torch.randn(1, 8, 2, 4)
    config = {
        "enabled": True,
        "sparsity": 0.75,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 1,
        "keep_sink": 1,
        "keep_near": 0,
    }

    actual = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=0, sparse_config=config
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_hsa_is_differentiable_and_keeps_shape():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "sparsity_base": 0.75,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
        "dense_current": True,
        "num_output_frames": 6,
        "num_frame_per_block": 1,
    }

    output = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=2, sparse_config=config
    )
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_hsa_query_block_batching_preserves_output():
    torch.manual_seed(2)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 24, 2, 4)
    v = torch.randn(1, 24, 2, 4)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
        "dense_current": True,
    }

    unbatched = hierarchical_sparse_attention(
        q,
        k,
        v,
        frame_seq=4,
        chunk_id=2,
        sparse_config={**config, "query_block_batch": 1},
    )
    batched = hierarchical_sparse_attention(
        q,
        k,
        v,
        frame_seq=4,
        chunk_id=2,
        sparse_config={**config, "query_block_batch": 2},
    )
    torch.testing.assert_close(batched, unbatched)


def test_hsa_rejects_misaligned_frame_blocks():
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 8, 1, 2)
    v = torch.randn(1, 8, 1, 2)
    config = {"enabled": True, "block_k": 3}

    try:
        hierarchical_sparse_attention(
            q, k, v, frame_seq=4, chunk_id=1, sparse_config=config
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected frame/block alignment validation to fail")


def test_hsa_rejects_sparse_current_chunk():
    try:
        SparseAttentionConfig.from_mapping(
            {"enabled": True, "dense_current": False, "keep_frames": 1}
        )
    except ValueError as error:
        assert "dense_current=true" in str(error)
    else:
        raise AssertionError("Expected sparse current-chunk routing to be rejected")
