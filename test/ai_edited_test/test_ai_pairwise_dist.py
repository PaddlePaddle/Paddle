# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.distance
# 自动生成的单测，覆盖 paddle.nn.layer.distance 模块中未覆盖的代码
# Target: paddle/nn/layer/distance.py

"""
测试模块：paddle.nn.layer.distance
Test Module: paddle.nn.layer.distance

本测试覆盖以下功能：
This test covers the following functions:
1. PairwiseDistance - 成对距离 / Pairwise distance with different p-norm, keepdim, epsilon
2. PairwiseDistance properties - eps/norm属性 / eps and norm property getters/setters
3. PairwiseDistance extra_repr - 字符串表示 / extra_repr method
"""

import unittest

import paddle
from paddle import nn


class TestPairwiseDistanceComprehensive(unittest.TestCase):
    """测试PairwiseDistance成对距离
    Test PairwiseDistance"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_pairwise_distance_p2(self):
        """测试p=2 (欧氏距离) / Test p=2 Euclidean distance"""
        pd = nn.PairwiseDistance(p=2.0)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        out = pd(x, y)
        self.assertEqual(out.shape, [2])

    def test_pairwise_distance_p1(self):
        """测试p=1 (曼哈顿距离) / Test p=1 Manhattan distance"""
        pd = nn.PairwiseDistance(p=1.0)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        out = pd(x, y)
        self.assertEqual(out.shape, [2])

    def test_pairwise_distance_p3(self):
        """测试p=3 / Test p=3"""
        pd = nn.PairwiseDistance(p=3.0)
        x = paddle.randn([2, 4])
        y = paddle.randn([2, 4])
        out = pd(x, y)
        self.assertEqual(out.shape, [2])

    def test_pairwise_distance_pinf(self):
        """测试p=inf (切比雪夫距离) / Test p=inf Chebyshev distance"""
        pd = nn.PairwiseDistance(p=float('inf'))
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        out = pd(x, y)
        self.assertEqual(out.shape, [2])

    def test_pairwise_distance_keepdim(self):
        """测试keepdim=True / Test with keepdim=True"""
        pd = nn.PairwiseDistance(p=2.0, keepdim=True)
        x = paddle.randn([2, 4])
        y = paddle.randn([2, 4])
        out = pd(x, y)
        self.assertEqual(out.shape, [2, 1])

    def test_pairwise_distance_epsilon(self):
        """测试自定义epsilon / Test with custom epsilon"""
        pd = nn.PairwiseDistance(p=2.0, epsilon=1e-4)
        x = paddle.randn([2, 4])
        y = paddle.randn([2, 4])
        out = pd(x, y)
        self.assertEqual(out.shape, [2])

    def test_pairwise_distance_1d_input(self):
        """测试1D输入 / Test with 1D input"""
        pd = nn.PairwiseDistance(p=2.0)
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        out = pd(x, y)
        self.assertEqual(out.shape, [])

    def test_pairwise_distance_1d_keepdim(self):
        """测试1D输入with keepdim / Test 1D input with keepdim=True"""
        pd = nn.PairwiseDistance(p=2.0, keepdim=True)
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        out = pd(x, y)
        self.assertEqual(out.shape, [1])

    def test_pairwise_distance_float64(self):
        """测试float64精度 / Test with float64 dtype"""
        pd = nn.PairwiseDistance(p=2.0)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float64')
        out = pd(x, y)
        self.assertEqual(out.dtype, paddle.float64)


class TestPairwiseDistanceProperties(unittest.TestCase):
    """测试PairwiseDistance属性
    Test PairwiseDistance properties"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_eps_property(self):
        """测试eps属性 / Test eps property"""
        pd = nn.PairwiseDistance(epsilon=1e-4)
        self.assertEqual(pd.eps, 1e-4)
        self.assertEqual(pd.epsilon, 1e-4)
        # Test setter
        pd.eps = 1e-3
        self.assertEqual(pd.epsilon, 1e-3)

    def test_norm_property(self):
        """测试norm属性 / Test norm property"""
        pd = nn.PairwiseDistance(p=3.0)
        self.assertEqual(pd.norm, 3.0)
        self.assertEqual(pd.p, 3.0)
        # Test setter
        pd.norm = 1.5
        self.assertEqual(pd.p, 1.5)


class TestPairwiseDistanceExtraRepr(unittest.TestCase):
    """测试PairwiseDistance extra_repr
    Test PairwiseDistance extra_repr"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_extra_repr_default(self):
        """测试默认extra_repr / Test default extra_repr"""
        pd = nn.PairwiseDistance()
        r = pd.extra_repr()
        self.assertIn('p=2.0', r)

    def test_extra_repr_custom(self):
        """测试自定义参数的extra_repr / Test extra_repr with custom params"""
        pd = nn.PairwiseDistance(p=3.0, epsilon=1e-4, keepdim=True, name='test')
        r = pd.extra_repr()
        self.assertIn('p=3.0', r)
        self.assertIn('epsilon', r)
        self.assertIn('keepdim', r)
        self.assertIn('name', r)


if __name__ == '__main__':
    unittest.main()
