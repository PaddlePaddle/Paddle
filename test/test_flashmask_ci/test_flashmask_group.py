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

import os
import unittest

import paddle
import paddle.distributed as dist
import paddle.nn.functional.flash_attention as fa_module
from paddle.nn.functional.flash_attention import (
    _flashmask_unique_id_cache,
    _get_or_create_unique_id,
    flashmask_attention,
)


def _is_distributed_env():
    return int(os.environ.get("PADDLE_TRAINERS_NUM", "1")) > 1


class _FakeGroup:
    """Minimal stand-in for dist.Group."""

    def __init__(self, gid, rank=0, nranks=1):
        self.id = gid
        self.rank = rank
        self.nranks = nranks


class TestGetOrCreateUniqueIdUnit(unittest.TestCase):
    def setUp(self):
        _flashmask_unique_id_cache.clear()
        self._orig_get_uid = fa_module.flashmask_get_unique_id
        self._orig_all_gather = dist.all_gather_object
        # replace flashmask_get_unique_id with a dummy func
        fa_module.flashmask_get_unique_id = lambda: paddle.ones(
            [128], dtype='uint8'
        )
        # replace all_gather_object with a dummy func
        dist.all_gather_object = lambda rl, obj, group=None: rl.append(obj)

    def tearDown(self):
        # recover the module functions
        fa_module.flashmask_get_unique_id = self._orig_get_uid
        dist.all_gather_object = self._orig_all_gather
        _flashmask_unique_id_cache.clear()

    def test_cache_hit(self):
        """cache hit → return (tensor, False)"""
        _flashmask_unique_id_cache[1] = paddle.ones([128], dtype='uint8')
        uid, is_new = _get_or_create_unique_id(_FakeGroup(1))
        self.assertFalse(is_new)
        self.assertEqual(uid.shape, [128])

    def test_cache_miss_rank0(self):
        """rank=0 → flashmask_get_unique_id() → cache → return True"""
        uid, is_new = _get_or_create_unique_id(_FakeGroup(2, rank=0))
        self.assertTrue(is_new)
        self.assertEqual(uid.shape, [128])
        self.assertEqual(uid.dtype, paddle.uint8)
        self.assertIn(2, _flashmask_unique_id_cache)

    def test_cache_miss_non_rank0(self):
        """rank!=0 → paddle.zeros → cache → return True"""
        uid, is_new = _get_or_create_unique_id(_FakeGroup(3, rank=1))
        self.assertTrue(is_new)
        self.assertTrue((uid.numpy() == 0).all())
        self.assertIn(3, _flashmask_unique_id_cache)


class TestFlashMaskAttentionGroupUnit(unittest.TestCase):
    """Cover group-related branches inside flashmask_attention on single GPU."""

    def setUp(self):
        _flashmask_unique_id_cache.clear()
        self._orig_get_uid = fa_module.flashmask_get_unique_id
        self._orig_all_gather = dist.all_gather_object
        fa_module.flashmask_get_unique_id = lambda: paddle.ones(
            [128], dtype='uint8'
        )
        dist.all_gather_object = lambda rl, obj, group=None: rl.append(obj)

    def tearDown(self):
        fa_module.flashmask_get_unique_id = self._orig_get_uid
        dist.all_gather_object = self._orig_all_gather
        _flashmask_unique_id_cache.clear()

    def test_group_none(self):
        q = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        k = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        v = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        out = flashmask_attention(q, k, v, causal=True)
        self.assertEqual(out.shape, [1, 128, 2, 64])

    def test_group_nranks1(self):
        q = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        k = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        v = paddle.rand([1, 128, 2, 64], dtype='bfloat16').cuda()
        out = flashmask_attention(
            q, k, v, causal=True, group=_FakeGroup(100, rank=0, nranks=1)
        )
        self.assertEqual(out.shape, [1, 128, 2, 64])


# Multi-GPU integration tests — only under paddle.distributed.launch
@unittest.skipUnless(
    _is_distributed_env(),
    "Requires multi-GPU (run with: python -m paddle.distributed.launch --gpus=0,1)",
)
class TestGetOrCreateUniqueIdDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def tearDown(self):
        _flashmask_unique_id_cache.clear()

    def test_first_call_returns_is_new_true(self):
        group = dist.new_group(list(range(dist.get_world_size())))
        uid, is_new = _get_or_create_unique_id(group)
        self.assertTrue(is_new)
        self.assertEqual(uid.shape, [128])
        self.assertEqual(uid.dtype, paddle.uint8)

    def test_second_call_returns_is_new_false(self):
        group = dist.new_group(list(range(dist.get_world_size())))
        uid1, is_new1 = _get_or_create_unique_id(group)
        uid2, is_new2 = _get_or_create_unique_id(group)
        self.assertTrue(is_new1)
        self.assertFalse(is_new2)
        self.assertTrue(paddle.equal_all(uid1, uid2).item())

    def test_unique_id_consistent_across_ranks(self):
        group = dist.new_group(list(range(dist.get_world_size())))
        uid, _ = _get_or_create_unique_id(group)
        uid_list = []
        dist.all_gather_object(uid_list, uid.numpy().tolist(), group=group)
        for i in range(1, len(uid_list)):
            self.assertEqual(uid_list[0], uid_list[i])


@unittest.skipUnless(
    _is_distributed_env(),
    "Requires multi-GPU (run with: python -m paddle.distributed.launch --gpus=0,1)",
)
class TestFlashMaskAttentionGroupParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def tearDown(self):
        _flashmask_unique_id_cache.clear()

    def test_group_none_no_distributed(self):
        q = paddle.rand([1, 8, 2, 32], dtype='bfloat16')
        k = paddle.rand([1, 8, 2, 32], dtype='bfloat16')
        v = paddle.rand([1, 8, 2, 32], dtype='bfloat16')
        out = flashmask_attention(q, k, v, causal=True)
        self.assertEqual(out.shape, [1, 8, 2, 32])

    def test_group_extracts_rank_nranks_for_mask_validation(self):
        world_size = dist.get_world_size()
        if world_size < 2:
            self.skipTest("Need at least 2 GPUs")
        group = dist.new_group(list(range(world_size)))
        B, S, H, D = 1, 8, 2, 32
        q = paddle.rand([B, S, H, D], dtype='bfloat16')
        k = paddle.rand([B, S, H, D], dtype='bfloat16')
        v = paddle.rand([B, S, H, D], dtype='bfloat16')
        startend_row_indices = paddle.full(
            [B, 1, S * world_size, 1], S * world_size, dtype='int32'
        )
        try:
            flashmask_attention(
                q, k, v, startend_row_indices, causal=True, group=group
            )
        except Exception as e:
            self.assertNotIn(
                "startend_row_indices.shape[2] must be equal to seqlen_k",
                str(e),
            )


if __name__ == '__main__':
    unittest.main()
