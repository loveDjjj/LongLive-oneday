from __future__ import annotations

import os
from contextlib import nullcontext

import torch


_TORCH_NPU = None
_TORCH_NPU_IMPORT_TRIED = False


def _load_torch_npu():
    global _TORCH_NPU, _TORCH_NPU_IMPORT_TRIED
    if _TORCH_NPU_IMPORT_TRIED:
        return _TORCH_NPU
    _TORCH_NPU_IMPORT_TRIED = True
    try:
        import torch_npu  # type: ignore
    except ImportError:
        _TORCH_NPU = None
    else:
        _TORCH_NPU = torch_npu
    return _TORCH_NPU


def npu_available() -> bool:
    _load_torch_npu()
    return hasattr(torch, "npu") and torch.npu.is_available()


def accelerator_type() -> str:
    requested = os.environ.get("LLV2_DEVICE", "auto").strip().lower()
    if requested in {"npu", "ascend"}:
        if not npu_available():
            raise RuntimeError("LLV2_DEVICE=npu was requested, but torch_npu/NPU is not available.")
        return "npu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("LLV2_DEVICE=cuda was requested, but CUDA is not available.")
        return "cuda"
    if requested not in {"", "auto"}:
        raise ValueError(f"Unsupported LLV2_DEVICE={requested!r}; use auto, cuda, or npu.")
    if npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_npu() -> bool:
    return accelerator_type() == "npu"


def is_cuda() -> bool:
    return accelerator_type() == "cuda"


def distributed_backend() -> str:
    override = os.environ.get("LLV2_DISTRIBUTED_BACKEND", "").strip()
    if override:
        return override
    return "hccl" if is_npu() else "nccl"


def set_device(local_rank: int) -> torch.device:
    kind = accelerator_type()
    if kind == "npu":
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    if kind == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def current_device():
    kind = accelerator_type()
    if kind == "npu":
        return torch.npu.current_device()
    if kind == "cuda":
        return torch.cuda.current_device()
    return torch.device("cpu")


def default_device(local_rank: int | None = None) -> torch.device:
    kind = accelerator_type()
    if kind == "npu":
        index = torch.npu.current_device() if local_rank is None else local_rank
        return torch.device(f"npu:{index}")
    if kind == "cuda":
        index = torch.cuda.current_device() if local_rank is None else local_rank
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def empty_cache() -> None:
    kind = accelerator_type()
    if kind == "npu":
        torch.npu.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


def ipc_collect() -> None:
    if is_cuda():
        torch.cuda.ipc_collect()


def synchronize(device=None) -> None:
    kind = accelerator_type()
    if kind == "npu":
        torch.npu.synchronize(device)
    elif kind == "cuda":
        torch.cuda.synchronize(device)


def _device_type(device) -> str:
    if hasattr(device, "type"):
        return str(device.type)
    return str(device).split(":", 1)[0]


def supports_streams(device) -> bool:
    kind = _device_type(device)
    accelerator = getattr(torch, kind, None)
    return (
        kind in {"cuda", "npu"}
        and accelerator is not None
        and hasattr(accelerator, "Stream")
        and hasattr(accelerator, "Event")
        and hasattr(accelerator, "stream")
    )


def create_stream(device):
    """Create a CUDA/NPU stream on ``device``."""
    kind = _device_type(device)
    accelerator = getattr(torch, kind, None)
    if not supports_streams(device):
        raise RuntimeError(f"{kind} runtime does not expose Stream/Event support")
    try:
        return accelerator.Stream(device=device)
    except TypeError:
        with device_context(device):
            return accelerator.Stream()


def create_event(device, *, enable_timing: bool = False):
    """Create an event using the runtime associated with ``device``."""
    kind = _device_type(device)
    accelerator = getattr(torch, kind, None)
    if not supports_streams(device):
        raise RuntimeError(f"{kind} runtime does not expose Stream/Event support")
    try:
        return accelerator.Event(enable_timing=enable_timing)
    except TypeError:
        return accelerator.Event()


def stream_context(stream, device=None):
    """Enter the CUDA/NPU stream context for ``stream``."""
    kind = _device_type(device if device is not None else stream.device)
    accelerator = getattr(torch, kind, None)
    if accelerator is None or not hasattr(accelerator, "stream"):
        raise RuntimeError(f"{kind} runtime does not expose a stream context")
    return accelerator.stream(stream)


def device_context(device):
    if device is None:
        return nullcontext()
    device = torch.device(device)
    if device.type == "npu":
        return torch.npu.device(device)
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if npu_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory_gb(device=None) -> float:
    kind = accelerator_type()
    if kind == "cuda":
        device = default_device() if device is None else torch.device(device)
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats["active_bytes.all.current"]
        bytes_reserved = memory_stats["reserved_bytes.all.current"]
        bytes_free, _ = torch.cuda.mem_get_info(device)
        return (bytes_free + bytes_reserved - bytes_active) / (1024 ** 3)
    if kind == "npu":
        device = default_device() if device is None else torch.device(device)
        try:
            bytes_free, _ = torch.npu.mem_get_info(device)
            return bytes_free / (1024 ** 3)
        except Exception:
            # torch_npu versions differ in memory-stat APIs. Return a large
            # value so low-memory CUDA-only offload paths do not trigger.
            return 64.0
    return 0.0


def total_memory_gb(device=None) -> float:
    kind = accelerator_type()
    if kind == "cuda":
        device = default_device() if device is None else torch.device(device)
        return torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if kind == "npu":
        try:
            device = default_device() if device is None else torch.device(device)
            _, total = torch.npu.mem_get_info(device)
            return total / (1024 ** 3)
        except Exception:
            return 64.0
    return 0.0
