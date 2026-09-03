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

# [AUTO-GENERATED]
# Target: paddle/nn/functional/input.py
# Coverage target: improve coverage for input functions (one_hot, embedding_renorm_,
#   embedding, embedding_with_scaled_gradient)
"""
Tests for paddle.nn.functional.input module.
测试 paddle.nn.functional.input 模块的单元测试。
"""

import unittest

import numpy as np

import paddle
from paddle.nn import functional as F


class TestOneHot(unittest.TestCase):
    """Tests for one_hot function. / one_hot 函数的测试。"""

    def test_one_hot_basic(self):
        """Test one_hot with explicit num_classes. / 测试显式 num_classes 的 one_hot。"""
        x = paddle.to_tensor([1, 2, 0, 3], dtype='int64')
        out = F.one_hot(x, num_classes=4)
        self.assertEqual(out.shape, [4, 4])
        self.assertEqual(out.dtype, paddle.float32)
        # Verify one-hot encoding
        result = out.numpy()
        np.testing.assert_allclose(result[0, 1], 1.0, rtol=1e-5)
        np.testing.assert_allclose(result[0, 0], 0.0, rtol=1e-5)

    def test_one_hot_auto_num_classes(self):
        """Test one_hot with auto num_classes (num_classes=-1). / 测试自动 num_classes 的 one_hot。"""
        x = paddle.to_tensor([0, 1, 2, 0], dtype='int64')
        out = F.one_hot(x, num_classes=-1)
        self.assertEqual(out.shape, [4, 3])
        self.assertEqual(out.dtype, paddle.float32)

    def test_one_hot_2d_input(self):
        """Test one_hot with 2D input. / 测试二维输入的 one_hot。"""
        x = paddle.to_tensor([[0, 1], [2, 1]], dtype='int64')
        out = F.one_hot(x, num_classes=3)
        self.assertEqual(out.shape, [2, 2, 3])

    def test_one_hot_alias_input(self):
        """Test one_hot with input alias. / 测试 input 别名的 one_hot。"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        out = F.one_hot(input=x, num_classes=2)
        self.assertEqual(out.shape, [2, 2])

    def test_one_hot_int32(self):
        """Test one_hot with int32 input. / 测试 int32 输入的 one_hot。"""
        x = paddle.to_tensor([0, 1, 0], dtype='int32')
        out = F.one_hot(x, num_classes=2)
        self.assertEqual(out.shape, [3, 2])


class TestEmbeddingRenum(unittest.TestCase):
    """Tests for embedding_renorm_ function. / embedding_renorm_ 函数的测试。"""

    def test_embedding_renorm_basic(self):
        """Test embedding_renorm_ with basic params. / 测试基本参数的 embedding_renorm_。"""
        x = paddle.to_tensor([0, 1, 2, 0], dtype='int64')
        weight = paddle.randn([4, 8], dtype='float32')
        weight_renormed = F.embedding_renorm_(
            x, weight, max_norm=1.0, norm_type=2.0
        )
        self.assertEqual(weight_renormed.shape, [4, 8])

    def test_embedding_renorm_l1(self):
        """Test embedding_renorm_ with L1 norm. / 测试 L1 范数的 embedding_renorm_。"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        weight = paddle.randn([4, 8], dtype='float32')
        weight_renormed = F.embedding_renorm_(
            x, weight, max_norm=2.0, norm_type=1.0
        )
        self.assertEqual(weight_renormed.shape, [4, 8])


class TestEmbedding(unittest.TestCase):
    """Tests for embedding function. / embedding 函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype='int64')
        self.weight = paddle.randn([10, 16], dtype='float32')

    def test_embedding_basic(self):
        """Test embedding with basic params. / 测试基本参数的 embedding。"""
        out = F.embedding(self.x, self.weight)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_with_padding_idx(self):
        """Test embedding with padding_idx. / 测试带 padding_idx 的 embedding。"""
        out = F.embedding(self.x, self.weight, padding_idx=0)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_negative_padding_idx(self):
        """Test embedding with negative padding_idx. / 测试负 padding_idx 的 embedding。"""
        out = F.embedding(self.x, self.weight, padding_idx=-1)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_sparse(self):
        """Test embedding with sparse=True. / 测试 sparse 的 embedding。"""
        out = F.embedding(self.x, self.weight, sparse=True)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_with_max_norm(self):
        """Test embedding with max_norm. / 测试带 max_norm 的 embedding。"""
        out = F.embedding(self.x, self.weight, max_norm=1.0)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_alias_input(self):
        """Test embedding with input alias. / 测试 input 别名的 embedding。"""
        out = F.embedding(input=self.x, weight=self.weight)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_embedding_invalid_padding_idx(self):
        """Test embedding raises ValueError for invalid padding_idx. / 测试无效 padding_idx 时 embedding 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.embedding(self.x, self.weight, padding_idx=20)

    def test_embedding_scale_grad_by_freq_error(self):
        """Test embedding raises error when scale_grad_by_freq and sparse conflict. / 测试 scale_grad_by_freq 和 sparse 冲突时抛出错误。"""
        with self.assertRaises(AttributeError):
            F.embedding(
                self.x, self.weight, scale_grad_by_freq=True, sparse=True
            )

    def test_embedding_scale_grad_by_freq(self):
        """Test embedding with scale_grad_by_freq. / 测试 scale_grad_by_freq 的 embedding。"""
        out = F.embedding(
            self.x, self.weight, scale_grad_by_freq=True, sparse=False
        )
        self.assertEqual(out.shape, [2, 3, 16])


if __name__ == '__main__':
    unittest.main()
