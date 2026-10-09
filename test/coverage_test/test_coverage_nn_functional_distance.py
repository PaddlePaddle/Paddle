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

# [AUTO-GENERATED] Test file for paddle.nn.functional.distance
# 覆盖模块: paddle/nn/functional/distance.py
# 未覆盖行: 95,96,97,99,102,105,106,107,108,111,112,113,119,120,124
# Covered module: paddle/nn/functional/distance.py
# Uncovered lines: 95,96,97,99,102,105,106,107,108,111,112,113,119,120,124

import unittest

import numpy as np

import paddle


class TestPairwiseDistance(unittest.TestCase):
    """测试 pairwise_distance 函数的各种参数组合
    Test pairwise_distance function with various parameter combinations"""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def test_pairwise_distance_basic(self):
        """测试基本的 pairwise_distance 计算
        Test basic pairwise_distance computation"""
        x = paddle.to_tensor([[1.0, 3.0], [3.0, 5.0]], dtype='float32')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(result.shape, [2])
        # pairwise distance with p=2: sqrt((x1-y1)^2 + (x2-y2)^2) + epsilon
        np.testing.assert_allclose(
            result.numpy(), [5.0, 5.0], rtol=1e-5, atol=1e-5
        )

    def test_pairwise_distance_p1(self):
        """测试 p=1 (曼哈顿距离) 时的 pairwise_distance
        Test pairwise_distance with p=1 (Manhattan distance)"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32')
        y = paddle.to_tensor([[4.0, 6.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y, p=1.0)
        # p=1: |1-4| + |2-6| = 7, but with epsilon added
        self.assertEqual(result.shape, [1])

    def test_pairwise_distance_keepdim(self):
        """测试 keepdim=True 的 pairwise_distance
        Test pairwise_distance with keepdim=True"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y, keepdim=True)
        self.assertEqual(result.shape, [2, 1])

    def test_pairwise_distance_1d_input(self):
        """测试一维输入的 pairwise_distance
        Test pairwise_distance with 1D input"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        y = paddle.to_tensor([4.0, 6.0], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(result.shape, [])

    def test_pairwise_distance_epsilon_zero(self):
        """测试 epsilon=0 时的 pairwise_distance
        Test pairwise_distance with epsilon=0"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32')
        y = paddle.to_tensor([[4.0, 6.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y, epsilon=0.0)
        self.assertEqual(result.shape, [1])

    def test_pairwise_distance_float64(self):
        """测试 float64 数据类型的 pairwise_distance
        Test pairwise_distance with float64 dtype"""
        x = paddle.to_tensor([[1.0, 3.0]], dtype='float64')
        y = paddle.to_tensor([[5.0, 6.0]], dtype='float64')
        result = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(result.dtype, paddle.float64)

    def test_pairwise_distance_alias_params(self):
        """测试使用参数别名的 pairwise_distance (x1, x2, eps)
        Test pairwise_distance with parameter aliases (x1, x2, eps)"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32')
        y = paddle.to_tensor([[4.0, 6.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x1=x, x2=y, eps=1e-6)
        self.assertEqual(result.shape, [1])

    def test_pairwise_distance_large_p(self):
        """测试大 p 值的 pairwise_distance
        Test pairwise_distance with large p value"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32')
        y = paddle.to_tensor([[4.0, 6.0]], dtype='float32')
        result = paddle.nn.functional.pairwise_distance(x, y, p=3.0)
        self.assertEqual(result.shape, [1])


class TestPdist(unittest.TestCase):
    """测试 pdist 函数
    Test pdist function"""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def test_pdist_basic(self):
        """测试基本的 pdist 计算
        Test basic pdist computation"""
        x = paddle.to_tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype='float32'
        )
        result = paddle.nn.functional.pdist(x, p=2.0)
        # N=3, output shape = 3*(3-1)/2 = 3
        self.assertEqual(result.shape, [3])

    def test_pdist_p1(self):
        """测试 p=1 的 pdist
        Test pdist with p=1"""
        x = paddle.to_tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype='float32'
        )
        result = paddle.nn.functional.pdist(x, p=1.0)
        self.assertEqual(result.shape, [3])
        # distances: |0-1|+|0-1|=2, |0-2|+|0-2|=4, |1-2|+|1-2|=2
        np.testing.assert_allclose(
            result.numpy(), [2.0, 4.0, 2.0], rtol=1e-5, atol=1e-5
        )

    def test_pdist_assert_2d(self):
        """测试 pdist 对非2D输入的断言
        Test pdist assertion for non-2D input"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        with self.assertRaises(AssertionError):
            paddle.nn.functional.pdist(x)

    def test_pdist_4_elements(self):
        """测试4个元素的 pdist (6对距离)
        Test pdist with 4 elements (6 pairs)"""
        paddle.seed(2023)
        a = paddle.randn([4, 5])
        result = paddle.nn.functional.pdist(a)
        # N=4, output shape = 4*3/2 = 6
        self.assertEqual(result.shape, [6])


if __name__ == '__main__':
    unittest.main()
