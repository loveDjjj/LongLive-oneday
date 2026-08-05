import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from utils.config import normalize_config, validate_hsa_cag_training_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "train" / "hsa_cag.yaml"


class TrainingConfigContractTest(unittest.TestCase):
    def _load_config_with_temporary_paths(self, root):
        model_root = root / "model"
        model_root.mkdir()
        train_prompts = root / "train.txt"
        eval_prompts = root / "eval.txt"
        generator = root / "generator.pt"
        for path in (train_prompts, eval_prompts, generator):
            path.touch()

        config = OmegaConf.load(CONFIG_PATH)
        config.model_kwargs.model_root = str(model_root)
        config.real_model_kwargs.model_root = str(model_root)
        config.fake_model_kwargs.model_root = str(model_root)
        config.checkpoints.generator_ckpt = str(generator)
        config.data.data_path = str(train_prompts)
        config.data.eval_data_path = str(eval_prompts)
        return normalize_config(config)

    def test_release_yaml_matches_maintained_training_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir)
            )

            validated = validate_hsa_cag_training_config(config)

            self.assertEqual(validated.sequence_parallel_size, 4)
            self.assertEqual(validated.sampling_steps, 4)
            self.assertEqual(validated.image_or_video_shape, [1, 32, 48, 44, 80])

    def test_sp_must_divide_heads_and_block_frames(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir)
            )
            config.sequence_parallel_size = 12

            with self.assertRaisesRegex(ValueError, "num_frame_per_block"):
                validate_hsa_cag_training_config(config)

    def test_sparse_blocks_must_divide_per_rank_tokens_for_sp8(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._load_config_with_temporary_paths(
                Path(temporary_dir)
            )
            config.sequence_parallel_size = 8
            config.model_kwargs.sparse_config.block_q = 80
            config.model_kwargs.sparse_config.block_k = 80

            with self.assertRaisesRegex(ValueError, "440 tokens per frame at SP8"):
                validate_hsa_cag_training_config(config)


if __name__ == "__main__":
    unittest.main()
