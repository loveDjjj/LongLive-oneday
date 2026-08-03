import random
import unittest
from unittest import mock

import numpy as np
import torch

from utils.dataset import ResumableDistributedSampler
from utils.training_state import (
    capture_rng_state,
    restore_fsdp_optimizer_state,
    restore_rng_state,
)


class ResumableDistributedSamplerTest(unittest.TestCase):
    def test_skips_resume_offset_only_on_first_iteration(self):
        dataset = list(range(12))
        sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
            start_index=2,
        )

        self.assertEqual(list(sampler), [4, 6, 8, 10])
        self.assertEqual(list(sampler), [0, 2, 4, 6, 8, 10])

    def test_uses_shared_seed_to_keep_rank_partitions_disjoint(self):
        dataset = list(range(12))
        rank0 = ResumableDistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=True, seed=42
        )
        rank1 = ResumableDistributedSampler(
            dataset, num_replicas=2, rank=1, shuffle=True, seed=42
        )

        rank0_indices = list(rank0)
        rank1_indices = list(rank1)
        self.assertEqual(set(rank0_indices) & set(rank1_indices), set())
        self.assertEqual(
            sorted(rank0_indices + rank1_indices),
            list(range(len(dataset))),
        )

    def test_rejects_negative_resume_offset(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ResumableDistributedSampler(
                list(range(4)),
                num_replicas=1,
                rank=0,
                start_index=-1,
            )


class FullTrainingStateTest(unittest.TestCase):
    def test_converts_and_restores_fsdp_optimizer_state(self):
        fsdp = mock.Mock()
        fsdp.optim_state_dict_to_load.return_value = {"state": "local"}
        model = object()
        optimizer = mock.Mock()

        restore_fsdp_optimizer_state(
            fsdp, model, optimizer, {"state": "full"}
        )

        fsdp.optim_state_dict_to_load.assert_called_once_with(
            model, optimizer, {"state": "full"}
        )
        optimizer.load_state_dict.assert_called_once_with({"state": "local"})

    def test_round_trips_python_numpy_and_torch_rng_state(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            saved_state = capture_rng_state()

        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)

        restore_rng_state(saved_state)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        torch.testing.assert_close(torch.rand(1), expected_torch)


if __name__ == "__main__":
    unittest.main()
