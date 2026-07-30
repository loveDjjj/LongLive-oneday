import unittest

from utils.parallel_layout import validate_sp_dp_layout
from third_party.aisbench_adapter.prepare_vbench_videos import VIDEO_PATTERN


class NativeWanParallelLayoutTest(unittest.TestCase):
    def test_all_twelve_card_head_divisible_layouts(self):
        for sp_size, dp_size in ((12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12)):
            self.assertEqual(
                validate_sp_dp_layout(
                    world_size=12,
                    sp_size=sp_size,
                    dp_size=dp_size,
                    num_heads=24,
                ),
                (sp_size, dp_size),
            )

    def test_rejects_world_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "parallel layout mismatch"):
            validate_sp_dp_layout(
                world_size=8, sp_size=2, dp_size=6, num_heads=24
            )

    def test_rejects_non_divisible_head_count(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            validate_sp_dp_layout(
                world_size=12, sp_size=12, dp_size=1, num_heads=10
            )

    def test_uneven_prompt_shards_cover_every_index_once(self):
        shards = [list(range(dp_rank, 186, 4)) for dp_rank in range(4)]
        self.assertEqual([len(shard) for shard in shards], [47, 47, 46, 46])
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(186)))

    def test_native_wan_filename_is_vbench_adapter_compatible(self):
        match = VIDEO_PATTERN.match("rank4-137-0_wan22_bf16_dp2_sp2.mp4")
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), ("137", "0"))


if __name__ == "__main__":
    unittest.main()
