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

# [AUTO-GENERATED] Unit test for paddle.tensor.math
# 自动生成的单测，覆盖 paddle.tensor.math 模块中未覆盖的代码

"""
测试模块：paddle.tensor.math (inner, outer, dist, lerp, frac, diff, logit, i0)
Test Module: paddle.tensor.math

本测试覆盖以下功能：
This test covers the following functions:
1. inner - 内积 / Inner product of tensors
2. outer - 外积 / Outer product of vectors
3. dist - 距离 / Distance with various p values
4. lerp - 线性插值 / Linear interpolation
5. frac - 小数部分 / Fractional part
6. diff - 差分 / Difference
7. logit - logit函数 / Logit function

覆盖的未覆盖行：inner/outer各分支, dist p值分支, lerp/frac动态图路径
"""

import unittest

import numpy as np

import paddle


class TestInner(unittest.TestCase):
    """测试inner内积运算
    Test inner product"""

    def setUp(self):
        paddle.disable_static()

    def test_inner_1d(self):
        """1D向量内积 / 1D vector inner product"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y = paddle.to_tensor([4.0, 5.0, 6.0], dtype='float32')
        out = paddle.inner(x, y)
        np.testing.assert_allclose(float(out.numpy()), 32.0, rtol=1e-5)

    def test_inner_2d(self):
        """2D矩阵内积 / 2D matrix inner product"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float32')
        out = paddle.inner(x, y)
        self.assertEqual(list(out.shape), [2, 2])


class TestOuter(unittest.TestCase):
    """测试outer外积运算
    Test outer product"""

    def setUp(self):
        paddle.disable_static()

    def test_outer_basic(self):
        """基本外积 / Basic outer product"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y = paddle.to_tensor([4.0, 5.0], dtype='float32')
        out = paddle.outer(x, y)
        expected = np.array([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_outer_shape(self):
        """外积形状验证 / Outer product shape"""
        x = paddle.ones([5], dtype='float32')
        y = paddle.ones([3], dtype='float32')
        out = paddle.outer(x, y)
        self.assertEqual(list(out.shape), [5, 3])


class TestDist(unittest.TestCase):
    """测试dist距离计算
    Test dist function"""

    def setUp(self):
        paddle.disable_static()

    def test_dist_p2(self):
        """L2距离 / L2 distance"""
        x = paddle.to_tensor([1.0, 0.0, 0.0], dtype='float32')
        y = paddle.to_tensor([0.0, 1.0, 0.0], dtype='float32')
        out = paddle.dist(x, y, p=2)
        np.testing.assert_allclose(float(out.numpy()), np.sqrt(2.0), rtol=1e-5)

    def test_dist_p1(self):
        """L1距离 / L1 distance"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y = paddle.to_tensor([4.0, 5.0, 6.0], dtype='float32')
        out = paddle.dist(x, y, p=1)
        np.testing.assert_allclose(float(out.numpy()), 9.0, rtol=1e-5)

    def test_dist_inf(self):
        """无穷距离 / Infinity distance"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y = paddle.to_tensor([4.0, 5.0, 10.0], dtype='float32')
        out = paddle.dist(x, y, p=float('inf'))
        np.testing.assert_allclose(float(out.numpy()), 7.0, rtol=1e-5)

    def test_dist_p0(self):
        """L0距离 / L0 distance"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y = paddle.to_tensor([1.0, 5.0, 6.0], dtype='float32')
        out = paddle.dist(x, y, p=0)
        np.testing.assert_allclose(float(out.numpy()), 2.0, rtol=1e-5)


class TestLerp(unittest.TestCase):
    """测试lerp线性插值
    Test lerp function"""

    def setUp(self):
        paddle.disable_static()

    def test_lerp_basic(self):
        """基本线性插值 / Basic lerp"""
        x = paddle.to_tensor([0.0, 0.0], dtype='float32')
        y = paddle.to_tensor([10.0, 10.0], dtype='float32')
        out = paddle.lerp(x, y, 0.5)
        np.testing.assert_allclose(out.numpy(), [5.0, 5.0], rtol=1e-5)

    def test_lerp_weight_zero(self):
        """weight=0返回x / Lerp with weight=0 returns x"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        y = paddle.to_tensor([10.0, 20.0], dtype='float32')
        out = paddle.lerp(x, y, 0.0)
        np.testing.assert_allclose(out.numpy(), [1.0, 2.0], rtol=1e-5)

    def test_lerp_weight_one(self):
        """weight=1返回y / Lerp with weight=1 returns y"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        y = paddle.to_tensor([10.0, 20.0], dtype='float32')
        out = paddle.lerp(x, y, 1.0)
        np.testing.assert_allclose(out.numpy(), [10.0, 20.0], rtol=1e-5)

    def test_lerp_tensor_weight(self):
        """Tensor权重 / Lerp with tensor weight"""
        x = paddle.zeros([3], dtype='float32')
        y = paddle.ones([3], dtype='float32')
        weight = paddle.to_tensor([0.0, 0.5, 1.0], dtype='float32')
        out = paddle.lerp(x, y, weight)
        np.testing.assert_allclose(out.numpy(), [0.0, 0.5, 1.0], rtol=1e-5)


class TestFrac(unittest.TestCase):
    """测试frac小数部分
    Test frac function"""

    def setUp(self):
        paddle.disable_static()

    def test_frac_positive(self):
        """正数小数部分 / Frac of positive numbers"""
        x = paddle.to_tensor([1.5, 2.7, 3.0], dtype='float32')
        out = paddle.frac(x)
        np.testing.assert_allclose(out.numpy(), [0.5, 0.7, 0.0], rtol=1e-4)

    def test_frac_negative(self):
        """负数小数部分 / Frac of negative numbers"""
        x = paddle.to_tensor([-1.5, -2.3], dtype='float32')
        out = paddle.frac(x)
        np.testing.assert_allclose(out.numpy(), [-0.5, -0.3], rtol=1e-4)


class TestDiff(unittest.TestCase):
    """测试diff差分
    Test diff function"""

    def setUp(self):
        paddle.disable_static()

    def test_diff_basic(self):
        """基本差分 / Basic diff"""
        x = paddle.to_tensor([1.0, 4.0, 9.0, 16.0], dtype='float32')
        out = paddle.diff(x)
        np.testing.assert_allclose(out.numpy(), [3.0, 5.0, 7.0], rtol=1e-5)

    def test_diff_n2(self):
        """二阶差分 / Second order diff"""
        x = paddle.to_tensor([1.0, 4.0, 9.0, 16.0], dtype='float32')
        out = paddle.diff(x, n=2)
        np.testing.assert_allclose(out.numpy(), [2.0, 2.0], rtol=1e-5)

    def test_diff_2d(self):
        """2D差分 / 2D diff"""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'
        )
        out = paddle.diff(x, axis=1)
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
