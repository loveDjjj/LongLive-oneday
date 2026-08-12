from unittest.mock import patch

import torch

from utils.wan_5b_wrapper import WanDiffusionWrapper
from wan_5b.modules.causal_model import CausalWanModel
from wan_5b.modules.causal_model_sp_ulysses import UlyssesSPCausalWanModel


def _tiny_model(model_cls, *, defer_sla_linear_init=False):
    return model_cls(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=1,
        num_layers=1,
        local_attn_size=4,
        sink_size=1,
        num_frame_per_block=1,
        defer_sla_linear_init=defer_sla_linear_init,
    )


def test_from_config_honors_deferred_sla_linear_construction():
    for model_cls in (CausalWanModel, UlyssesSPCausalWanModel):
        direct = _tiny_model(model_cls)
        deferred = model_cls.from_config(
            direct.config, defer_sla_linear_init=True
        )
        assert deferred.blocks[0].self_attn.sla_linear is None

        deferred.initialize_sla_linear()
        linear = deferred.blocks[0].self_attn.sla_linear
        assert linear.weight.device.type == "cpu"
        assert torch.count_nonzero(linear.weight) == 0
        assert torch.count_nonzero(linear.bias) == 0


def test_direct_construction_keeps_zero_initialized_sla_linear():
    for model_cls in (CausalWanModel, UlyssesSPCausalWanModel):
        model = _tiny_model(model_cls)
        linear = model.blocks[0].self_attn.sla_linear
        assert torch.count_nonzero(linear.weight) == 0
        assert torch.count_nonzero(linear.bias) == 0


def test_wrapper_materializes_layer_and_resets_saved_config():
    loaded = _tiny_model(CausalWanModel, defer_sla_linear_init=True)
    with patch.object(CausalWanModel, "from_pretrained", return_value=loaded) as load:
        wrapper = WanDiffusionWrapper(model_root="/tmp/wan", is_causal=True)

    assert load.call_args.kwargs["defer_sla_linear_init"] is True
    assert wrapper.model.blocks[0].self_attn.sla_linear is not None
    assert wrapper.model.config.defer_sla_linear_init is False
