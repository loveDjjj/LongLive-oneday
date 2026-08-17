import torch
import pytest

from wan_5b.modules.causal_model_sp_ulysses import UlyssesCausalWanSelfAttention


def _tokens(values):
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1)


def _rolling_cache():
    return {
        "k": _tokens([0, 1, 2, 3]),
        "v": _tokens([10, 11, 12, 13]),
        "global_end_index": torch.tensor([4]),
        "local_end_index": torch.tensor([4]),
        "pinned_start": torch.tensor([-1]),
        "pinned_len": torch.tensor([0]),
    }


def test_inference_revisits_current_chunk_after_rolling_cache():
    attention = UlyssesCausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=4,
        sink_size=1,
    )
    cache = _rolling_cache()

    with torch.no_grad():
        attention._update_cache_and_get_kv(
            _tokens([4, 5]),
            _tokens([14, 15]),
            cache,
            current_start=4,
            current_end=6,
            frame_seqlen=1,
        )
        k_full, v_full, kv_layout = attention._update_cache_and_get_kv(
            _tokens([40, 50]),
            _tokens([140, 150]),
            cache,
            current_start=4,
            current_end=6,
            frame_seqlen=1,
        )

    assert cache["global_end_index"].item() == 6
    assert cache["local_end_index"].item() == 4
    assert cache["k"].flatten().tolist() == [0, 3, 40, 50]
    assert cache["v"].flatten().tolist() == [10, 13, 140, 150]
    assert k_full.flatten().tolist() == [0, 3, 40, 50]
    assert v_full.flatten().tolist() == [10, 13, 140, 150]
    assert kv_layout.history_frames == 2
    assert kv_layout.current_frames == 2
    assert kv_layout.global_sink_frames == (0,)


def test_autograd_recompute_still_rejects_a_rolling_cache():
    attention = UlyssesCausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=4,
        sink_size=1,
    )
    cache = _rolling_cache()

    with torch.no_grad():
        attention._update_cache_and_get_kv(
            _tokens([4, 5]),
            _tokens([14, 15]),
            cache,
            current_start=4,
            current_end=6,
            frame_seqlen=1,
        )

    with pytest.raises(RuntimeError, match="autograd checkpoint recomputation"):
        attention._update_cache_and_get_kv(
            _tokens([40, 50]).requires_grad_(),
            _tokens([140, 150]).requires_grad_(),
            cache,
            current_start=4,
            current_end=6,
            frame_seqlen=1,
        )


def test_multishot_sink_metadata_follows_assembled_resident_kv():
    attention = UlyssesCausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=4,
        sink_size=1,
    )
    attention.global_sink_size = 1
    attention.max_attention_size = 4
    cache = {
        "k": _tokens(range(8)),
        "v": _tokens(range(10, 18)),
        "global_end_index": torch.tensor([8]),
        "local_end_index": torch.tensor([8]),
        "pinned_start": torch.tensor([2]),
        "pinned_len": torch.tensor([1]),
    }

    with torch.no_grad():
        k_full, _, kv_layout = attention._update_cache_and_get_kv(
            _tokens([60, 70]),
            _tokens([160, 170]),
            cache,
            current_start=6,
            current_end=8,
            frame_seqlen=1,
        )

    assert k_full.flatten().tolist() == [0, 2, 60, 70]
    assert kv_layout.history_frames == 2
    assert kv_layout.current_frames == 2
    assert kv_layout.global_sink_frames == (0,)
    assert kv_layout.shot_sink_frames == (1,)
