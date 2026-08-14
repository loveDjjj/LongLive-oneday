import sys
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from utils.lora_utils import set_peft_model_state_dict_for_ulysses


class _ModelWithHfTpMetadata(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.projection._hf_tp_plan = object()
        self.projection._hf_device_mesh = object()


def test_peft_load_temporarily_disables_hf_tp_metadata():
    model = _ModelWithHfTpMetadata()
    original_plan = model.projection._hf_tp_plan
    original_mesh = model.projection._hf_device_mesh
    state_dict = {"adapter": torch.ones(1)}
    result = object()

    save_and_load = ModuleType("peft.utils.save_and_load")

    def tp_shard(*_args, **_kwargs):
        raise ImportError("EmbeddingParallel is unavailable")

    save_and_load._maybe_shard_state_dict_for_tp = tp_shard

    def load(current_model, current_state):
        assert current_model is model
        assert current_state is state_dict
        assert model.projection._hf_tp_plan is None
        assert model.projection._hf_device_mesh is None
        save_and_load._maybe_shard_state_dict_for_tp(
            current_model, current_state, "default"
        )
        return result

    fake_peft = SimpleNamespace(set_peft_model_state_dict=load)
    fake_peft_utils = ModuleType("peft.utils")
    fake_peft_utils.save_and_load = save_and_load
    modules = {
        "peft": fake_peft,
        "peft.utils": fake_peft_utils,
        "peft.utils.save_and_load": save_and_load,
    }
    with mock.patch.dict(sys.modules, modules):
        assert set_peft_model_state_dict_for_ulysses(model, state_dict) is result

    assert model.projection._hf_tp_plan is original_plan
    assert model.projection._hf_device_mesh is original_mesh
    assert save_and_load._maybe_shard_state_dict_for_tp is tp_shard


def test_peft_load_restores_hf_tp_metadata_after_failure():
    model = _ModelWithHfTpMetadata()
    original_plan = model.projection._hf_tp_plan
    original_mesh = model.projection._hf_device_mesh

    save_and_load = ModuleType("peft.utils.save_and_load")

    def tp_shard(*_args, **_kwargs):
        return None

    save_and_load._maybe_shard_state_dict_for_tp = tp_shard

    def fail(*_args, **_kwargs):
        raise RuntimeError("adapter load failed")

    fake_peft = SimpleNamespace(set_peft_model_state_dict=fail)
    fake_peft_utils = ModuleType("peft.utils")
    fake_peft_utils.save_and_load = save_and_load
    modules = {
        "peft": fake_peft,
        "peft.utils": fake_peft_utils,
        "peft.utils.save_and_load": save_and_load,
    }
    with mock.patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match="adapter load failed"):
            set_peft_model_state_dict_for_ulysses(model, {})

    assert model.projection._hf_tp_plan is original_plan
    assert model.projection._hf_device_mesh is original_mesh
    assert save_and_load._maybe_shard_state_dict_for_tp is tp_shard
