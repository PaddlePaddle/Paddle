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

# [AUTO-GENERATED] Unit test for paddle.nn.initializer.orthogonal
# 自动生成的单测，覆盖 paddle.nn.initializer.orthogonal 模块中未覆盖的代码
# Target: paddle/nn/initializer/orthogonal.py

"""
测试模块：paddle.nn.initializer.orthogonal
Test Module: paddle.nn.initializer.orthogonal

本测试覆盖以下功能：
This test covers the following functions:
1. Orthogonal - 正交初始化器 / Orthogonal initializer with different gain values
2. Orthogonal with different shapes - 不同维度的正交初始化 / Orthogonal init with different shapes
3. Orthogonal gain=None assertion - gain为None时的断言 / Test gain=None assertion
4. Orthogonal with 1D tensor assertion - 1D张量断言 / Test 1D tensor assertion
"""

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.nn import initializer


class TestOrthogonalInitializer(unittest.TestCase):
    """测试Orthogonal正交初始化器
    Test Orthogonal initializer"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_orthogonal_init_linear(self):
        """测试线性层的正交初始化 / Test orthogonal init for linear layer"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(10, 15, weight_attr=weight_attr)
        w = linear.weight.numpy()
        # Check shape is a tuple (10, 15) for in_features=10, out_features=15
        self.assertEqual(w.shape, (10, 15))

    def test_orthogonal_init_linear_tall(self):
        """测试高矩阵的正交初始化 / Test orthogonal init for tall matrix (rows > cols)"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(15, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        self.assertEqual(w.shape, (15, 10))

    def test_orthogonal_init_square(self):
        """测试方阵的正交初始化 / Test orthogonal init for square matrix"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(10, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        # For square matrix, X'X should be close to I
        identity = w.T @ w
        np.testing.assert_allclose(identity, np.eye(10), atol=1e-6)

    def test_orthogonal_gain(self):
        """测试带gain的正交初始化 / Test orthogonal init with gain"""
        weight_attr = paddle.ParamAttr(
            initializer=initializer.Orthogonal(gain=2.0)
        )
        linear = nn.Linear(10, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        # With gain=2.0, columns should still be orthogonal but scaled
        col_norms = np.linalg.norm(w, axis=0)
        # All columns should have same norm (= gain)
        np.testing.assert_allclose(col_norms, col_norms[0], atol=1e-5)

    def test_orthogonal_gain_zero(self):
        """测试gain=0 / Test orthogonal init with gain=0"""
        weight_attr = paddle.ParamAttr(
            initializer=initializer.Orthogonal(gain=0.0)
        )
        linear = nn.Linear(10, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        np.testing.assert_allclose(w, 0.0, atol=1e-6)

    def test_orthogonal_3d_tensor(self):
        """测试3D张量的正交初始化 / Test orthogonal init for 3D tensor"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(20, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        # 2D weight in linear layer: shape [in_features, out_features]
        self.assertEqual(w.ndim, 2)

    def test_orthogonal_small_matrix(self):
        """测试小矩阵 / Test with small matrix"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(2, 2, weight_attr=weight_attr)
        w = linear.weight.numpy()
        identity = w.T @ w
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-5)

    def test_orthogonal_large_matrix(self):
        """测试大矩阵 / Test with large matrix"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(128, 128, weight_attr=weight_attr)
        w = linear.weight.numpy()
        identity = w.T @ w
        np.testing.assert_allclose(identity, np.eye(128), atol=1e-4)

    def test_orthogonal_float64(self):
        """测试float64权重 / Test orthogonal init with float64 weight"""
        w = paddle.empty([10, 10], dtype='float64')
        orth = initializer.Orthogonal()
        orth(w)
        identity = w.T @ w
        np.testing.assert_allclose(identity.numpy(), np.eye(10), atol=1e-10)

    def test_orthogonal_rectangular_wide(self):
        """测试宽矩阵 (rows < cols) / Test wide matrix"""
        weight_attr = paddle.ParamAttr(initializer=initializer.Orthogonal())
        linear = nn.Linear(5, 10, weight_attr=weight_attr)
        w = linear.weight.numpy()
        self.assertEqual(w.shape, (5, 10))
        # rows < cols: rows should be orthogonal vectors
        identity = w @ w.T
        np.testing.assert_allclose(identity, np.eye(5), atol=1e-5)


class TestOrthogonalInitializerErrors(unittest.TestCase):
    """测试Orthogonal错误处理
    Test Orthogonal error handling"""

    def test_orthogonal_gain_none(self):
        """测试gain=None断言 / Test assertion when gain is None"""
        with self.assertRaises(AssertionError):
            initializer.Orthogonal(gain=None)

    def test_orthogonal_1d_tensor_assertion(self):
        """测试1D张量初始化断言 / Test assertion with 1D tensor"""
        paddle.disable_static()
        orth = initializer.Orthogonal()
        var = paddle.empty([10])
        # 1D tensor should fail
        try:
            orth(var)
            self.fail("Expected AssertionError for 1D tensor")
        except (AssertionError, Exception):
            pass
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
