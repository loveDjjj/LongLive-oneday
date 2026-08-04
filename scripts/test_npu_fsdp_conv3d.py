#!/usr/bin/env python3
"""Isolate the Wan 5B patch Conv3d under the configured FSDP strategy."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.distributed import fsdp_wrap, launch_distributed_job


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            48, 3072, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

    def forward(self, x):
        return self.patch_embedding(x)


def log(message):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[conv3d-smoke][rank={rank}] {message}", flush=True)


def synchronize():
    torch.npu.synchronize()


def run_forward(module, x, label, backward):
    start = time.perf_counter()
    output = module(x)
    synchronize()
    log(
        f"{label} forward passed in {time.perf_counter() - start:.2f}s; "
        f"input={tuple(x.shape)} output={tuple(output.shape)} dtype={output.dtype}"
    )
    if backward:
        start = time.perf_counter()
        output.float().square().mean().backward()
        synchronize()
        log(f"{label} backward passed in {time.perf_counter() - start:.2f}s")
    del output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sharding-strategy",
        choices=("full", "hybrid_full", "hybrid_zero2", "no_shard"),
        default="hybrid_full",
    )
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    launch_distributed_job()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(1234)

    log(
        f"initialized world={world_size} local_rank={local_rank} "
        f"device={torch.npu.current_device()} strategy={args.sharding_strategy}"
    )

    x = torch.randn(
        1, 48, 8, 44, 80, device=f"npu:{local_rank}", dtype=torch.bfloat16
    )

    plain = PatchEmbedding().to(device=f"npu:{local_rank}", dtype=torch.bfloat16)
    run_forward(plain, x, "plain", args.backward)
    del plain
    torch.npu.empty_cache()
    dist.barrier()

    fsdp_module = PatchEmbedding().to(
        device=f"npu:{local_rank}", dtype=torch.bfloat16
    )
    fsdp_module = fsdp_wrap(
        fsdp_module,
        sharding_strategy=args.sharding_strategy,
        mixed_precision=True,
        wrap_strategy="size",
    )
    run_forward(fsdp_module, x, "fsdp", args.backward)
    dist.barrier()

    if rank == 0:
        print("NPU FSDP Conv3d smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
