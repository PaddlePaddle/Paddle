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

# [AUTO-GENERATED] Test file for paddle.nn.functional.input
# 覆盖模块: paddle/nn/functional/input.py
# 未覆盖行: 118,119,121,122,124,125,127,128,129,130,137,315,316,318,324,326,332
# Covered module: paddle/nn/functional/input.py
# Uncovered lines: 118,119,121,122,124,125,127,128,129,130,137,315,316,318,324,326,332

import unittest

import numpy as np

import paddle


class TestOneHot(unittest.TestCase):
    """测试 one_hot 函数
    Test one_hot function"""

    def test_one_hot_basic(self):
        """测试基本的 one_hot
        Test basic one_hot"""
        label = paddle.to_tensor([1, 1, 3, 0], dtype='int64')
        result = paddle.nn.functional.one_hot(label, num_classes=4)
        self.assertEqual(result.shape, [4, 4])
        self.assertEqual(result.dtype, paddle.float32)
        expected = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_one_hot_int32(self):
        """测试 int32 输入的 one_hot
        Test one_hot with int32 input"""
        label = paddle.to_tensor([0, 1, 2], dtype='int32')
        result = paddle.nn.functional.one_hot(label, num_classes=3)
        self.assertEqual(result.shape, [3, 3])

    def test_one_hot_auto_num_classes(self):
        """测试自动推断 num_classes 的 one_hot (num_classes=-1)
        Test one_hot with auto num_classes (num_classes=-1)"""
        label = paddle.to_tensor([1, 0, 2], dtype='int64')
        result = paddle.nn.functional.one_hot(label, num_classes=-1)
        # max(label) + 1 = 3
        self.assertEqual(result.shape, [3, 3])

    def test_one_hot_2d_input(self):
        """测试2D输入的 one_hot
        Test one_hot with 2D input"""
        label = paddle.to_tensor([[0, 1], [2, 0]], dtype='int64')
        result = paddle.nn.functional.one_hot(label, num_classes=3)
        self.assertEqual(result.shape, [2, 2, 3])

    def test_one_hot_alias_input(self):
        """测试使用 input 别名的 one_hot
        Test one_hot with 'input' alias parameter"""
        label = paddle.to_tensor([1, 0], dtype='int64')
        result = paddle.nn.functional.one_hot(input=label, num_classes=2)
        self.assertEqual(result.shape, [2, 2])

    def test_one_hot_3d_input(self):
        """测试3D输入的 one_hot
        Test one_hot with 3D input"""
        label = paddle.to_tensor([[[0, 1]]], dtype='int64')
        result = paddle.nn.functional.one_hot(label, num_classes=2)
        self.assertEqual(result.shape, [1, 1, 2, 2])


class TestEmbedding(unittest.TestCase):
    """测试 embedding 函数
    Test embedding function"""

    def test_embedding_basic(self):
        """测试基本的 embedding
        Test basic embedding"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        weight = paddle.randn([10, 4])
        result = paddle.nn.functional.embedding(x, weight)
        self.assertEqual(result.shape, [3, 4])

    def test_embedding_2d_indices(self):
        """测试2D索引的 embedding
        Test embedding with 2D indices"""
        x = paddle.arange(3, 6).reshape((3, 1)).astype(paddle.int64)
        weight = paddle.full(shape=(10, 3), fill_value=2.0).astype(
            paddle.float32
        )
        result = paddle.nn.functional.embedding(x, weight, sparse=True)
        self.assertEqual(result.shape, [3, 1, 3])
        np.testing.assert_allclose(result.numpy(), np.full((3, 1, 3), 2.0))

    def test_embedding_padding_idx(self):
        """测试带 padding_idx 的 embedding
        Test embedding with padding_idx"""
        x = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
        weight = paddle.randn([5, 4])
        result = paddle.nn.functional.embedding(x, weight, padding_idx=3)
        self.assertEqual(result.shape, [4, 4])
        # padding_idx=3 的位置应该全为0
        np.testing.assert_array_equal(result[3].numpy(), np.zeros(4))

    def test_embedding_negative_padding_idx(self):
        """测试负 padding_idx 的 embedding
        Test embedding with negative padding_idx"""
        x = paddle.to_tensor([0, 1, 5], dtype='int64')
        weight = paddle.randn([6, 4])
        result = paddle.nn.functional.embedding(x, weight, padding_idx=-1)
        self.assertEqual(result.shape, [3, 4])
        # padding_idx=-1 means last row (index 5)
        np.testing.assert_array_equal(result[2].numpy(), np.zeros(4))

    def test_embedding_invalid_padding_idx(self):
        """测试无效 padding_idx 的报错
        Test embedding raises error for invalid padding_idx"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        weight = paddle.randn([5, 4])
        with self.assertRaises(ValueError):
            paddle.nn.functional.embedding(x, weight, padding_idx=10)

    def test_embedding_sparse(self):
        """测试 sparse 模式的 embedding
        Test embedding in sparse mode"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        weight = paddle.randn([10, 4])
        result = paddle.nn.functional.embedding(x, weight, sparse=True)
        self.assertEqual(result.shape, [3, 4])

    def test_embedding_max_norm(self):
        """测试带 max_norm 的 embedding (会 renorm weight)
        Test embedding with max_norm (renorms weight)"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        weight = paddle.randn([5, 4]) * 10  # large values
        result = paddle.nn.functional.embedding(x, weight, max_norm=1.0)
        self.assertEqual(result.shape, [2, 4])

    def test_embedding_alias_input(self):
        """测试使用 input 别名的 embedding
        Test embedding with 'input' alias parameter"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        weight = paddle.randn([5, 4])
        result = paddle.nn.functional.embedding(input=x, weight=weight)
        self.assertEqual(result.shape, [2, 4])


class TestEmbeddingRenorm(unittest.TestCase):
    """测试 embedding_renorm_ 函数
    Test embedding_renorm_ function"""

    def test_embedding_renorm_basic(self):
        """测试基本的 embedding_renorm_
        Test basic embedding_renorm_"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        weight = paddle.randn([5, 4]) * 10
        original_norms = paddle.norm(weight[:3], p=2, axis=1)
        result = paddle.nn.functional.embedding_renorm_(x, weight, max_norm=1.0)
        new_norms = paddle.norm(result[:3], p=2, axis=1)
        # All norms should be <= max_norm
        self.assertTrue(paddle.all(new_norms <= 1.0 + 1e-5).item())

    def test_embedding_renorm_no_change(self):
        """测试 norm 已经小于 max_norm 时不会改变 weight
        Test embedding_renorm_ doesn't change weight when norms are already small"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        weight = paddle.randn([5, 4]) * 0.01  # small values
        result = paddle.nn.functional.embedding_renorm_(
            x, weight, max_norm=100.0
        )
        # Should not change since norms are already small
        np.testing.assert_allclose(result.numpy(), weight.numpy(), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
