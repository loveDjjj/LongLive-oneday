import sys
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

    def load(current_model, current_state):
        assert current_model is model
        assert current_state is state_dict
        assert model.projection._hf_tp_plan is None
        assert model.projection._hf_device_mesh is None
        return result

    fake_peft = SimpleNamespace(set_peft_model_state_dict=load)
    with mock.patch.dict(sys.modules, {"peft": fake_peft}):
        assert set_peft_model_state_dict_for_ulysses(model, state_dict) is result

    assert model.projection._hf_tp_plan is original_plan
    assert model.projection._hf_device_mesh is original_mesh


def test_peft_load_restores_hf_tp_metadata_after_failure():
    model = _ModelWithHfTpMetadata()
    original_plan = model.projection._hf_tp_plan
    original_mesh = model.projection._hf_device_mesh

    def fail(*_args, **_kwargs):
        raise RuntimeError("adapter load failed")

    fake_peft = SimpleNamespace(set_peft_model_state_dict=fail)
    with mock.patch.dict(sys.modules, {"peft": fake_peft}):
        with pytest.raises(RuntimeError, match="adapter load failed"):
            set_peft_model_state_dict_for_ulysses(model, {})

    assert model.projection._hf_tp_plan is original_plan
    assert model.projection._hf_device_mesh is original_mesh
