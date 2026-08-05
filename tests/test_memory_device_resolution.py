import importlib
from unittest import mock

import torch

import utils.device
import utils.memory


def test_import_does_not_resolve_accelerator_device():
    with mock.patch.object(
        utils.device,
        "default_device",
        side_effect=AssertionError("accelerator resolved during import"),
    ):
        importlib.reload(utils.memory)
    importlib.reload(utils.memory)


def test_default_memory_device_is_resolved_lazily():
    with (
        mock.patch.object(
            utils.memory,
            "default_device",
            return_value=torch.device("cpu"),
        ) as resolve_device,
        mock.patch.object(utils.memory, "free_memory_gb", return_value=12.5) as free_memory,
    ):
        assert utils.memory.get_cuda_free_memory_gb() == 12.5

    resolve_device.assert_called_once_with()
    free_memory.assert_called_once_with(torch.device("cpu"))
