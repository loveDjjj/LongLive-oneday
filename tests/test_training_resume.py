import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from utils.dataset import ResumableDistributedSampler
from utils.training_state import (
    build_generator_linear_sidecar,
    capture_rng_state,
    find_latest_training_checkpoint,
    list_training_checkpoints,
    resume_samples_per_rank,
    restore_fsdp_optimizer_state,
    restore_rng_state,
    should_save_final_checkpoint,
    validate_sparse_checkpoint_method,
)


def test_builds_portable_generator_linear_sidecar():
    linear = {"blocks.0.self_attn.sla_linear.weight": torch.ones(2, 2)}
    full = {
        "generator_linear": linear,
        "step": 10,
        "generator_train_scope": "linear_only",
        "generator_trainable_parameters": list(linear),
        "sparse_method": "hsa_sla_cag",
        "world_size": 12,
        "sequence_parallel_size": 4,
        "data_parallel_size": 3,
        "critic_lora": {"large": torch.ones(1)},
        "generator_optimizer": {"large": torch.ones(1)},
    }

    sidecar = build_generator_linear_sidecar(full)

    assert sidecar["generator_linear"] is linear
    assert sidecar["checkpoint_format"] == "longlive_generator_linear_v1"
    assert "critic_lora" not in sidecar
    assert "generator_optimizer" not in sidecar


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

    def test_cycle_can_resume_at_a_later_shuffle_epoch(self):
        dataset = list(range(8))
        sampler = ResumableDistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, seed=9
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, sampler=sampler
        )

        from utils.dataset import cycle

        resumed = next(cycle(dataloader, start_epoch=3)).tolist()
        sampler.set_epoch(3)
        self.assertEqual(resumed, list(sampler))


class TrainingCursorTest(unittest.TestCase):
    def test_v3_cursor_uses_logical_dp_samples(self):
        samples, metadata = resume_samples_per_rank(
            {
                "world_size": 12,
                "sequence_parallel_size": 4,
                "data_parallel_size": 3,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "global_samples_consumed": 4800,
            },
            step=100,
            current_data_parallel_size=3,
            current_sequence_parallel_size=4,
            current_batch_size=1,
            current_accumulation_steps=16,
        )
        self.assertEqual(samples, 1600)
        self.assertEqual(metadata["remainder"], 0)

    def test_v2_cursor_ignores_sp_overcount(self):
        samples, _ = resume_samples_per_rank(
            {
                "world_size": 12,
                "sequence_parallel_size": 4,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "global_samples_consumed": 19200,
            },
            step=100,
            current_data_parallel_size=3,
            current_sequence_parallel_size=4,
            current_batch_size=1,
            current_accumulation_steps=16,
        )
        self.assertEqual(samples, 1600)

    def test_v2_cursor_infers_missing_sp_from_current_layout(self):
        samples, metadata = resume_samples_per_rank(
            {
                "world_size": 12,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "global_samples_consumed": 19200,
            },
            step=100,
            current_data_parallel_size=3,
            current_sequence_parallel_size=4,
            current_batch_size=1,
            current_accumulation_steps=16,
        )
        self.assertEqual(samples, 1600)
        self.assertTrue(metadata["inferred_sequence_parallel_size"])

    def test_cursor_redistributes_when_dp_size_changes(self):
        samples, _ = resume_samples_per_rank(
            {"data_parallel_size": 3, "global_samples_consumed": 4800},
            step=100,
            current_data_parallel_size=6,
            current_sequence_parallel_size=4,
            current_batch_size=1,
            current_accumulation_steps=16,
        )
        self.assertEqual(samples, 800)


class CheckpointDiscoveryTest(unittest.TestCase):
    def test_prefers_current_layout_for_duplicate_step(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            legacy = root / "checkpoint_model_10" / "model.pt"
            current = root / "checkpoints" / "step_0000010" / "train_state.pt"
            later = root / "checkpoints" / "step_0000020" / "train_state.pt"
            for path in (legacy, current, later):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            checkpoints = list_training_checkpoints(root)

            self.assertEqual([item[0] for item in checkpoints], [10, 20])
            self.assertEqual(checkpoints[0][3], str(current))
            self.assertEqual(find_latest_training_checkpoint(root), str(later))


class FinalCheckpointTest(unittest.TestCase):
    def test_saves_final_step_when_it_is_outside_periodic_cadence(self):
        self.assertTrue(
            should_save_final_checkpoint(
                start_step=0, final_step=1001, save_interval=10, no_save=False
            )
        )

    def test_does_not_duplicate_periodic_or_unchanged_checkpoint(self):
        for values in (
            dict(start_step=0, final_step=1000, save_interval=10, no_save=False),
            dict(start_step=1000, final_step=1000, save_interval=10, no_save=False),
            dict(start_step=0, final_step=1001, save_interval=10, no_save=True),
        ):
            self.assertFalse(should_save_final_checkpoint(**values))


class SparseCheckpointContractTest(unittest.TestCase):
    def test_accepts_matching_sla_checkpoint(self):
        validate_sparse_checkpoint_method(
            {"sparse_method": "sla_cag"}, "sla_cag"
        )

    def test_rejects_legacy_or_hsa_checkpoint(self):
        for checkpoint in ({}, {"sparse_method": "hsa_cag"}):
            with self.assertRaisesRegex(ValueError, "expected sla_cag"):
                validate_sparse_checkpoint_method(checkpoint, "sla_cag")

    def test_accepts_matching_hsa_checkpoint(self):
        validate_sparse_checkpoint_method(
            {"sparse_method": "hsa_cag"}, "hsa_cag"
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
