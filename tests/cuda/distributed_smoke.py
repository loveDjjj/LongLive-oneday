#!/usr/bin/env python3
"""不加载 5B 权重的 CUDA 多卡准入：Ulysses 反向、混合精度 FSDP 和恢复。

使用 torchrun --standalone 启动；默认要求 4 张 CUDA 卡。
完整模型的显存、速度和 checkpoint 仍需另行验证。
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.distributed import fsdp_wrap, launch_distributed_job, promote_trainable_parameters
from wan_5b.distributed.sp_training import all_to_all_with_grad
from wan_5b.modules.sla_attention import SLAAttentionConfig, _projected_linear_attention


class TinyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, dtype=torch.bfloat16)
        self.sla_linear = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, value):
        hidden = self.proj(value)
        return hidden + _projected_linear_attention(
            hidden, hidden, hidden, history_tokens=0, chunk_id=1,
            config=SLAAttentionConfig(), cache=None, projection=self.sla_linear,
        )


def check_ulysses(device):
    world, rank = dist.get_world_size(), dist.get_rank()
    whole = torch.arange(1 * 16 * (2 * world) * 8, device=device, dtype=torch.float32)
    whole = whole.reshape(1, 16, 2 * world, 8) / 1000
    local = whole.chunk(world, dim=1)[rank].clone().requires_grad_()
    exchanged = all_to_all_with_grad(local, scatter_dim=2, gather_dim=1)
    torch.testing.assert_close(exchanged, whole.chunk(world, dim=2)[rank])
    exchanged.square().sum().backward()
    torch.testing.assert_close(local.grad, local.detach() * 2)
    restored = all_to_all_with_grad(exchanged.detach(), scatter_dim=1, gather_dim=2)
    torch.testing.assert_close(restored, local.detach())


def check_fsdp(device):
    import peft

    torch.manual_seed(73)
    model = peft.get_peft_model(
        TinyAdapterModel(), peft.LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
    )
    for name, parameter in model.named_parameters():
        if ".sla_linear." in name:
            parameter.requires_grad_(True)
    promote_trainable_parameters(model)
    wrapped = fsdp_wrap(
        model, sharding_strategy="hybrid_full", mixed_precision=True,
        separate_trainable_parameters=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in wrapped.parameters() if p.requires_grad], lr=2e-5,
    )
    value = torch.randn(1, 8, 2, 8, device=device, dtype=torch.bfloat16)
    # Replicated input and initial parameters give an exact multi-rank check
    # after FSDP reduction, without relying on any training-data sharding.
    dist.broadcast(value, src=0)
    before = {
        name: parameter.detach().clone()
        for name, parameter in wrapped.named_parameters() if parameter.requires_grad
    }
    loss = wrapped(value).float().square().mean()
    loss.backward()
    grad_squared = torch.zeros((), device=device)
    for parameter in wrapped.parameters():
        if parameter.requires_grad and parameter.numel():
            assert parameter.dtype == torch.float32, "trainable master lost FP32 storage"
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            grad_squared += parameter.grad.float().square().sum()
    dist.all_reduce(grad_squared)
    assert grad_squared.item() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    change = torch.zeros((), device=device)
    for name, parameter in wrapped.named_parameters():
        if name in before:
            change += (parameter.detach() - before[name]).abs().sum()
    dist.all_reduce(change)
    assert change.item() > 0, "optimizer did not update FP32 adapters"

    # Every rank serializes its own optimizer shard. All ranks participate in
    # the full-model gather, then reload the same state into the existing tree.
    full_state = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, full_state):
        state = {name: value.clone() for name, value in wrapped.state_dict().items()}
    buffer = io.BytesIO()
    torch.save({"model": state, "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False, map_location="cpu")
    expected = wrapped(value).detach()
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, full_state):
        wrapped.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(wrapped(value).detach(), expected, rtol=0, atol=0)
    # Verify optimizer slots remain usable after deserializing to CPU and
    # letting load_state_dict map them back to their parameter devices.
    wrapped(value).float().square().mean().backward()
    optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("此测试需要 CUDA GPU 和 torchrun")
    launch_distributed_job(backend="nccl")
    try:
        if dist.get_world_size() != args.world_size or 16 % args.world_size:
            raise ValueError("实际进程数必须匹配 --world-size，且整除 16")
        device = torch.device("cuda", torch.cuda.current_device())
        check_ulysses(device)
        check_fsdp(device)
        torch.cuda.synchronize(device)
        if dist.get_rank() == 0:
            print("ulysses_backward=passed mixed_dtype_fsdp=passed optimizer_resume=passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
