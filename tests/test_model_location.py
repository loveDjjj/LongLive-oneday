import unittest

from omegaconf import OmegaConf

from utils.config import resolve_model_location


class ModelLocationTest(unittest.TestCase):
    def test_resolves_shared_root_for_training_components(self):
        kwargs = OmegaConf.create({
            "model_name": "Wan2.2-TI2V-5B",
            "model_root": "/mnt/models/wan22",
            "local_attn_size": 32,
        })

        self.assertEqual(
            resolve_model_location(kwargs),
            ("Wan2.2-TI2V-5B", "/mnt/models/wan22"),
        )

    def test_preserves_legacy_defaults(self):
        self.assertEqual(
            resolve_model_location(None),
            ("Wan2.2-TI2V-5B", None),
        )


if __name__ == "__main__":
    unittest.main()
