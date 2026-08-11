from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def validate_sparse_checkpoint_method(checkpoint, expected_method):
    """Prevent resuming optimizer/LoRA state from another sparse algorithm."""
    actual_method = checkpoint.get("sparse_method")
    if actual_method != expected_method:
        actual_label = actual_method if actual_method is not None else "unlabeled/legacy"
        raise ValueError(
            f"checkpoint sparse method is {actual_label}, expected {expected_method}; "
            "start a new run directory or provide a matching checkpoint"
        )


def list_training_checkpoints(output_dir):
    """Return one checkpoint per step, preferring the current layout."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    checkpoints = {}
    for checkpoint_file in output_dir.glob("checkpoint_model_*/model.pt"):
        try:
            step = int(checkpoint_file.parent.name.removeprefix("checkpoint_model_"))
        except ValueError:
            continue
        checkpoints[step] = (
            step,
            str(checkpoint_file.parent),
            checkpoint_file.parent.name,
            str(checkpoint_file),
        )

    for checkpoint_file in (output_dir / "checkpoints").glob("step_*/train_state.pt"):
        try:
            step = int(checkpoint_file.parent.name.removeprefix("step_"))
        except ValueError:
            continue
        checkpoints[step] = (
            step,
            str(checkpoint_file.parent),
            checkpoint_file.parent.name,
            str(checkpoint_file),
        )

    return [checkpoints[step] for step in sorted(checkpoints)]


def find_latest_training_checkpoint(output_dir):
    checkpoints = list_training_checkpoints(output_dir)
    return checkpoints[-1][3] if checkpoints else None


def resume_samples_per_rank(
    checkpoint,
    *,
    step,
    current_data_parallel_size,
    current_sequence_parallel_size,
    current_batch_size,
    current_accumulation_steps,
):
    """Convert a global checkpoint cursor into samples consumed by each DP rank."""
    saved_world_size = int(
        checkpoint.get(
            "world_size",
            current_data_parallel_size * current_sequence_parallel_size,
        )
    )
    inferred_sp_size = "sequence_parallel_size" not in checkpoint
    saved_sp_size = int(
        checkpoint.get("sequence_parallel_size", current_sequence_parallel_size)
    )
    saved_dp_size = int(
        checkpoint.get(
            "data_parallel_size",
            saved_world_size // max(saved_sp_size, 1),
        )
    )
    saved_batch_size = int(checkpoint.get("batch_size", current_batch_size))
    saved_accumulation = int(
        checkpoint.get(
            "gradient_accumulation_steps", current_accumulation_steps
        )
    )

    # Version-2 checkpoints counted SP workers as independent samples. Rebuild
    # their cursor from the saved logical DP layout instead of trusting it.
    if "data_parallel_size" in checkpoint:
        global_samples = int(
            checkpoint.get(
                "global_samples_consumed",
                step * saved_dp_size * saved_batch_size * saved_accumulation,
            )
        )
    else:
        global_samples = (
            step * saved_dp_size * saved_batch_size * saved_accumulation
        )

    denominator = current_data_parallel_size * current_batch_size
    samples, remainder = divmod(global_samples, denominator)
    metadata = {
        "global_samples": global_samples,
        "remainder": remainder,
        "saved_world_size": saved_world_size,
        "saved_data_parallel_size": saved_dp_size,
        "saved_sequence_parallel_size": saved_sp_size,
        "inferred_sequence_parallel_size": inferred_sp_size,
        "saved_batch_size": saved_batch_size,
        "saved_accumulation_steps": saved_accumulation,
    }
    return samples * current_batch_size, metadata


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
