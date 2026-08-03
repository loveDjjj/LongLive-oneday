from __future__ import annotations

import random

import numpy as np
import torch


def restore_fsdp_optimizer_state(fsdp_cls, model, optimizer, full_state_dict):
    local_state_dict = fsdp_cls.optim_state_dict_to_load(
        model, optimizer, full_state_dict
    )
    optimizer.load_state_dict(local_state_dict)


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state().cpu(),
    }
    if hasattr(torch, "npu") and torch.npu.is_available():
        state["accelerator"] = torch.npu.get_rng_state().cpu()
        state["accelerator_type"] = "npu"
    elif torch.cuda.is_available():
        state["accelerator"] = torch.cuda.get_rng_state().cpu()
        state["accelerator_type"] = "cuda"
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy(),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])

    accelerator_state = state.get("accelerator")
    if accelerator_state is None:
        return
    if state.get("accelerator_type") == "npu" and hasattr(torch, "npu"):
        torch.npu.set_rng_state(accelerator_state)
    elif state.get("accelerator_type") == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(accelerator_state)
