import unittest

from wan_5b.distributed.sp_training import (
    build_sp_dp_rank_layout,
    resolve_kv_cache_heads,
)


class HsaCagSpDpLayoutTest(unittest.TestCase):
    def test_sp4_dp4_rank_layout(self):
        sp_groups, dp_groups = build_sp_dp_rank_layout(16, 4)

        self.assertEqual(
            sp_groups,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        )
        self.assertEqual(
            dp_groups,
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
        )

    def test_sp4_shards_24_attention_heads(self):
        self.assertEqual(resolve_kv_cache_heads(24, 4), 6)

    def test_invalid_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            build_sp_dp_rank_layout(12, 5)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            resolve_kv_cache_heads(24, 5)


if __name__ == "__main__":
    unittest.main()
