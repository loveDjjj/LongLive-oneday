"""Prompt-only datasets used by maintained SLA+CAG training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "


class PromptDataset(Dataset):
    """Read one UTF-8 prompt per non-empty line and repeat it per time block."""

    def __init__(
        self,
        data_path: str,
        num_blocks: int,
    ):
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.data_path = str(path)
        self.num_blocks = int(num_blocks)
        with path.open(encoding="utf-8") as handle:
            self._prompts = [line.strip() for line in handle if line.strip()]
        if not self._prompts:
            raise ValueError(f"Prompt file is empty: {path}")

    @property
    def mode(self):
        return "txt"

    @property
    def _mode(self):
        """Compatibility for existing progress output."""
        return self.mode

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, index):
        prompt = self._prompts[index]
        return {
            "idx": index,
            "prompt": prompt,
            "prompts": [prompt] * self.num_blocks,
        }


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that skips an initial per-rank sample offset once."""

    def __init__(self, *args, start_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        if start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {start_index}")
        self.start_index = int(start_index) % max(self.num_samples, 1)

    def __iter__(self):
        indices = list(super().__iter__())
        start_index = self.start_index
        self.start_index = 0
        return iter(indices[start_index:])

    def __len__(self):
        return max(super().__len__() - self.start_index, 0)


def prompt_collate_fn(batch):
    return {
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "prompt": [item["prompt"] for item in batch],
        "prompts": [item["prompts"] for item in batch],
    }


def cycle(dataloader, start_epoch=0):
    """Iterate forever and advance DistributedSampler's shuffle epoch."""
    if start_epoch < 0:
        raise ValueError(f"start_epoch must be non-negative, got {start_epoch}")
    epoch = int(start_epoch)
    while True:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1
