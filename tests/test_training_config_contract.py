import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from utils.config import normalize_config, validate_sla_cag_training_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "sla_cag": ROOT / "configs" / "train" / "sla_cag.yaml",
    "hsa_cag": ROOT / "configs" / "train" / "hsa_cag.yaml",
    "hsa_sla_cag": ROOT / "configs" / "train" / "hsa_sla_cag.yaml",
}


class TrainingConfigContractTest(unittest.TestCase):
    def _load_config_with_temporary_paths(self, root, method="sla_cag"):
        model_root = root / "model"
        model_root.mkdir()
        train_prompts = root / "train.txt"
        eval_prompts = root / "eval.txt"
        generator = root / "generator.pt"
        for path in (train_prompts, eval_prompts, generator):
            path.touch()

        config = OmegaConf.load(CONFIG_PATHS[method])
        config.model_kwargs.model_root = str(model_root)
        config.real_model_kwargs.model_root = str(model_root)
        config.fake_model_kwargs.model_root = str(model_root)
        config.checkpoints.generator_ckpt = str(generator)
        config.data.data_path = str(train_prompts)
        config.data.eval_data_path = str(eval_prompts)
        return normalize_config(config)

    def test_release_yaml_matches_maintained_training_contract(self):
        for method in CONFIG_PATHS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary_dir:
                config = self._load_config_with_temporary_paths(
                    Path(temporary_dir), method
                )

                validated = validate_sla_cag_training_config(config)

                expected_sp = 8 if method in {"sla_cag", "hsa_sla_cag"} else 4
                self.assertEqual(validated.sequence_parallel_size, expected_sp)
                self.assertEqual(validated.sampling_steps, 4)
                self.assertEqual(validated.image_or_video_shape, [1, 32, 48, 44, 80])
                self.assertEqual(validated.model_kwargs.sparse_config.method, method)
                expected_scope = (
                    "lora_plus_linear"
                    if method in {"sla_cag", "hsa_sla_cag"}
                    else "lora"
                )
                self.assertEqual(validated.generator_train_scope, expected_scope)
                if method in {"sla_cag", "hsa_sla_cag"}:
                    self.assertEqual(validated.lr_linear, 2.0e-5)

                if method in {"sla_cag", "hsa_sla_cag"}:
                    self.assertEqual(validated.gradient_accumulation_steps, 4)
                    self.assertEqual(validated.log_iters, 20)
                    self.assertEqual(validated.max_checkpoints, 5)
                    self.assertEqual(validated.max_iters, 200)
                    self.assertEqual(validated.dfake_gen_update_ratio, 1)
                if method == "hsa_sla_cag":
                    sparse = validated.model_kwargs.sparse_config
                    self.assertEqual(sparse.max_global_sink_frames, 2)
                    self.assertEqual(sparse.max_shot_sink_frames, 2)

    def test_hybrid_linear_only_override_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir), "hsa_sla_cag"
            )
            config.generator_train_scope = "linear_only"

            validated = validate_sla_cag_training_config(config)

            self.assertEqual(validated.generator_train_scope, "linear_only")

    def test_sla_rejects_legacy_lora_only_scope(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir), "sla_cag"
            )
            config.generator_train_scope = "lora"

            with self.assertRaisesRegex(ValueError, "lora_plus_linear"):
                validate_sla_cag_training_config(config)

    def test_sp_must_divide_heads_and_block_frames(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir)
            )
            config.sequence_parallel_size = 12

            with self.assertRaisesRegex(ValueError, "num_frame_per_block"):
                validate_sla_cag_training_config(config)

    def test_sla_blocks_use_post_exchange_sequence_for_sp8(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir)
            )
            config.sequence_parallel_size = 8

            validated = validate_sla_cag_training_config(config)

            self.assertEqual(validated.sequence_parallel_size, 8)


if __name__ == "__main__":
    unittest.main()
