# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flashmask_attention group parameter.

Run with:
    python -m paddle.distributed.launch --gpus=0,1 test_flashmask_group.py
"""

import unittest

import paddle
import paddle.distributed as dist
from paddle.nn.functional.flash_attention import (
    _flashmask_unique_id_cache,
    _get_or_create_unique_id,
    flashmask_attention,
)


class TestGetOrCreateUniqueId(unittest.TestCase):
    """Tests for the _get_or_create_unique_id cache mechanism."""

    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def tearDown(self):
        _flashmask_unique_id_cache.clear()

    def test_first_call_returns_is_new_true(self):
        """First call for a group should return is_new=True."""
        group = dist.new_group(list(range(dist.get_world_size())))
        uid, is_new = _get_or_create_unique_id(group)

        self.assertTrue(is_new)
        self.assertEqual(uid.shape, [128])
        self.assertEqual(uid.dtype, paddle.uint8)

    def test_second_call_returns_is_new_false(self):
        """Second call for the same group should return cached result."""
        group = dist.new_group(list(range(dist.get_world_size())))

        uid1, is_new1 = _get_or_create_unique_id(group)
        uid2, is_new2 = _get_or_create_unique_id(group)

        self.assertTrue(is_new1)
        self.assertFalse(is_new2)
        self.assertTrue(paddle.equal_all(uid1, uid2).item())

    def test_cache_keyed_by_group_id(self):
        """Cache should be keyed by group.id (int)."""
        group = dist.new_group(list(range(dist.get_world_size())))
        _get_or_create_unique_id(group)

        self.assertIn(group.id, _flashmask_unique_id_cache)

    def test_different_groups_cached_separately(self):
        """Different groups (different id) should have independent cache entries."""
        world_size = dist.get_world_size()
        group1 = dist.new_group(list(range(world_size)))
        group2 = dist.new_group(list(range(world_size)))

        _get_or_create_unique_id(group1)
        _get_or_create_unique_id(group2)

        self.assertIn(group1.id, _flashmask_unique_id_cache)
        self.assertIn(group2.id, _flashmask_unique_id_cache)
        self.assertNotEqual(group1.id, group2.id)

    def test_unique_id_consistent_across_ranks(self):
        """All ranks in the group should receive the same unique_id."""
        group = dist.new_group(list(range(dist.get_world_size())))
        uid, _ = _get_or_create_unique_id(group)

        uid_list = []
        dist.all_gather_object(uid_list, uid.numpy().tolist(), group=group)

        for i in range(1, len(uid_list)):
            self.assertEqual(
                uid_list[0],
                uid_list[i],
                f"Rank 0 and rank {i} got different unique_ids",
            )


class TestFlashMaskAttentionGroupParam(unittest.TestCase):
    """Tests for the group parameter in flashmask_attention."""

    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def tearDown(self):
        _flashmask_unique_id_cache.clear()

    def test_group_none_no_distributed(self):
        """group=None should work as non-distributed (rank=0, nranks=1)."""
        q = paddle.rand([1, 8, 2, 32], dtype='bfloat16')
        k = paddle.rand([1, 8, 2, 32], dtype='bfloat16')
        v = paddle.rand([1, 8, 2, 32], dtype='bfloat16')

        out = flashmask_attention(q, k, v, causal=True)
        self.assertEqual(out.shape, [1, 8, 2, 32])

    def test_group_extracts_rank_nranks_for_mask_validation(self):
        """When group is provided, nranks from group should be used in mask shape validation.

        In context parallel, startend_row_indices seqlen dim can be seqlen_k * nranks.
        Without group (nranks=1), a mask with seqlen = S * world_size would fail.
        With group (nranks=world_size), it should pass the shape assertion.
        """
        world_size = dist.get_world_size()
        if world_size < 2:
            self.skipTest("Need at least 2 GPUs")

        group = dist.new_group(list(range(world_size)))

        B, S, H, D = 1, 8, 2, 32
        q = paddle.rand([B, S, H, D], dtype='bfloat16')
        k = paddle.rand([B, S, H, D], dtype='bfloat16')
        v = paddle.rand([B, S, H, D], dtype='bfloat16')

        # Mask with seqlen = S * world_size (context parallel scenario)
        startend_row_indices = paddle.full(
            [B, 1, S * world_size, 1], S * world_size, dtype='int32'
        )

        try:
            flashmask_attention(
                q, k, v, startend_row_indices, causal=True, group=group
            )
        except Exception as e:
            # The mask shape assertion should pass thanks to group.nranks.
            # Other errors (NVSHMEM not compiled, fa_version, etc.) are expected.
            self.assertNotIn(
                "startend_row_indices.shape[2] must be equal to seqlen_k",
                str(e),
                "group.nranks should have been used for mask shape validation",
            )


if __name__ == '__main__':
    unittest.main()
