# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Test file for paddle.nn.functional.extension
# 覆盖模块: paddle/nn/functional/extension.py
# 未覆盖行: 130,131,133,134,135,136,137,139,141,145,146,231,232,233,236,238,244,323,324,330,331,333,335,336,338,348
# Covered module: paddle/nn/functional/extension.py
# Uncovered lines: 130,131,133,134,135,136,137,139,141,145,146,231,232,233,236,238,244,323,324,330,331,333,335,336,338,348

import unittest

import numpy as np

import paddle


class TestSequenceMask(unittest.TestCase):
    """测试 sequence_mask 函数
    Test sequence_mask function"""

    def test_sequence_mask_basic(self):
        """测试基本的 sequence_mask
        Test basic sequence_mask"""
        lengths = paddle.to_tensor([10, 9, 8])
        mask = paddle.nn.functional.sequence_mask(lengths)
        self.assertEqual(mask.shape, [3, 10])
        self.assertEqual(mask.dtype, paddle.int64)

    def test_sequence_mask_with_maxlen(self):
        """测试指定 maxlen 的 sequence_mask
        Test sequence_mask with specified maxlen"""
        lengths = paddle.to_tensor([3, 1, 1, 0])
        mask = paddle.nn.functional.sequence_mask(lengths, maxlen=4)
        self.assertEqual(mask.shape, [4, 4])
        expected = np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_sequence_mask_dtype_float32(self):
        """测试 float32 输出类型的 sequence_mask
        Test sequence_mask with float32 output dtype"""
        lengths = paddle.to_tensor([2, 3])
        mask = paddle.nn.functional.sequence_mask(lengths, dtype='float32')
        self.assertEqual(mask.dtype, paddle.float32)

    def test_sequence_mask_dtype_int32(self):
        """测试 int32 输出类型的 sequence_mask
        Test sequence_mask with int32 output dtype"""
        lengths = paddle.to_tensor([2, 3])
        mask = paddle.nn.functional.sequence_mask(lengths, dtype='int32')
        self.assertEqual(mask.dtype, paddle.int32)

    def test_sequence_mask_dtype_bool(self):
        """测试 bool 输出类型的 sequence_mask
        Test sequence_mask with bool output dtype"""
        lengths = paddle.to_tensor([2, 3])
        mask = paddle.nn.functional.sequence_mask(lengths, dtype='bool')
        self.assertEqual(mask.dtype, paddle.bool)

    def test_sequence_mask_2d_input(self):
        """测试2D输入的 sequence_mask
        Test sequence_mask with 2D input"""
        lengths = paddle.to_tensor([[2, 3], [1, 4]])
        mask = paddle.nn.functional.sequence_mask(lengths, maxlen=5)
        self.assertEqual(mask.shape, [2, 2, 5])

    def test_sequence_mask_no_maxlen(self):
        """测试不指定 maxlen 的 sequence_mask (使用 max(x))
        Test sequence_mask without maxlen (uses max(x))"""
        lengths = paddle.to_tensor([2, 5, 3])
        mask = paddle.nn.functional.sequence_mask(lengths)
        self.assertEqual(mask.shape, [3, 5])


class TestGatherTree(unittest.TestCase):
    """测试 gather_tree 函数
    Test gather_tree function"""

    def test_gather_tree_basic(self):
        """测试基本的 gather_tree
        Test basic gather_tree"""
        ids = paddle.to_tensor(
            [[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]
        )
        parents = paddle.to_tensor(
            [[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]
        )
        result = paddle.nn.functional.gather_tree(ids, parents)
        self.assertEqual(result.shape, [3, 2, 2])
        expected = np.array(
            [
                [[2, 2], [1, 6]],
                [[3, 3], [6, 1]],
                [[0, 1], [9, 0]],
            ]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_gather_tree_not_3d(self):
        """测试非3D输入的 gather_tree 报错
        Test gather_tree raises error for non-3D input"""
        ids = paddle.to_tensor([[1, 2], [3, 4]])
        parents = paddle.to_tensor([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            paddle.nn.functional.gather_tree(ids, parents)

    def test_gather_tree_shape_mismatch(self):
        """测试 ids 和 parents 形状不匹配时报错
        Test gather_tree raises error when ids and parents shapes differ"""
        ids = paddle.to_tensor([[[1, 2]]])
        parents = paddle.to_tensor([[[1, 2], [3, 4]]])
        with self.assertRaises(ValueError):
            paddle.nn.functional.gather_tree(ids, parents)

    def test_gather_tree_int32(self):
        """测试 int32 类型的 gather_tree
        Test gather_tree with int32 dtype"""
        ids = paddle.to_tensor(
            [[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]],
            dtype='int32',
        )
        parents = paddle.to_tensor(
            [[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]],
            dtype='int32',
        )
        result = paddle.nn.functional.gather_tree(ids, parents)
        self.assertEqual(result.shape, [3, 2, 2])


class TestTemporalShift(unittest.TestCase):
    """测试 temporal_shift 函数
    Test temporal_shift function"""

    def test_temporal_shift_basic(self):
        """测试基本的 temporal_shift
        Test basic temporal_shift"""
        x = paddle.randn([6, 4, 2, 2])
        out = paddle.nn.functional.temporal_shift(
            x, seg_num=2, shift_ratio=0.25
        )
        self.assertEqual(out.shape, [6, 4, 2, 2])

    def test_temporal_shift_nhwc(self):
        """测试 NHWC 格式的 temporal_shift
        Test temporal_shift with NHWC format"""
        x = paddle.randn([6, 2, 2, 4])
        out = paddle.nn.functional.temporal_shift(
            x, seg_num=2, shift_ratio=0.25, data_format='NHWC'
        )
        self.assertEqual(out.shape, [6, 2, 2, 4])

    def test_temporal_shift_invalid_format(self):
        """测试无效的 data_format 报错
        Test temporal_shift raises error for invalid data_format"""
        x = paddle.randn([6, 4, 2, 2])
        with self.assertRaises(ValueError):
            paddle.nn.functional.temporal_shift(
                x, seg_num=2, shift_ratio=0.25, data_format='INVALID'
            )

    def test_temporal_shift_different_seg_num(self):
        """测试不同 seg_num 的 temporal_shift
        Test temporal_shift with different seg_num"""
        x = paddle.randn([9, 8, 3, 3])
        out = paddle.nn.functional.temporal_shift(
            x, seg_num=3, shift_ratio=0.125
        )
        self.assertEqual(out.shape, [9, 8, 3, 3])

    def test_temporal_shift_float64(self):
        """测试 float64 类型的 temporal_shift
        Test temporal_shift with float64 dtype"""
        x = paddle.randn([6, 4, 2, 2], dtype='float64')
        out = paddle.nn.functional.temporal_shift(
            x, seg_num=2, shift_ratio=0.25
        )
        self.assertEqual(out.dtype, paddle.float64)


class TestDiagEmbed(unittest.TestCase):
    """测试 diag_embed 函数 (deprecated)
    Test diag_embed function (deprecated)"""

    def test_diag_embed_basic(self):
        """测试基本的 diag_embed
        Test basic diag_embed"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.nn.functional.diag_embed(x)
        self.assertEqual(result.shape, [3, 3])

    def test_diag_embed_with_offset(self):
        """测试带 offset 的 diag_embed
        Test diag_embed with offset"""
        x = paddle.to_tensor([1.0, 2.0])
        result = paddle.nn.functional.diag_embed(x, offset=1)
        self.assertEqual(result.shape, [3, 3])


if __name__ == '__main__':
    unittest.main()
