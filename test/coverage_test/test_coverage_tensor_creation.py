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

# [AUTO-GENERATED] Unit test for paddle.tensor.creation
# 自动生成的单测，覆盖 paddle.tensor.creation 模块中未覆盖的代码

"""
测试模块：paddle.tensor.creation (linspace, logspace, arange, full, meshgrid)
Test Module: paddle.tensor.creation

本测试覆盖以下功能：
This test covers the following functions:
1. linspace - 线性等分 / Linear space generation
2. logspace - 对数等分 / Log space generation
3. arange - 等差序列 / Arithmetic sequence generation
4. full - 全值填充 / Full fill tensor
5. meshgrid - 网格生成 / Mesh grid generation

覆盖的未覆盖行：linspace各分支, logspace各分支, arange边界条件
"""

import unittest

import numpy as np

import paddle


class TestLinspace(unittest.TestCase):
    """测试linspace线性等分
    Test linspace function"""

    def setUp(self):
        paddle.disable_static()

    def test_linspace_basic(self):
        """基本linspace / Basic linspace"""
        out = paddle.linspace(0, 10, 5)
        expected = np.linspace(0, 10, 5).astype('float32')
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_linspace_float64(self):
        """float64类型 / Linspace with float64"""
        out = paddle.linspace(0.0, 1.0, 11, dtype='float64')
        expected = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-10)

    def test_linspace_single_point(self):
        """单点linspace / Single point linspace"""
        out = paddle.linspace(5, 5, 1)
        np.testing.assert_allclose(out.numpy(), [5.0], rtol=1e-5)

    def test_linspace_reverse(self):
        """逆序linspace / Reverse linspace"""
        out = paddle.linspace(10, 0, 6)
        expected = np.linspace(10, 0, 6).astype('float32')
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


class TestLogspace(unittest.TestCase):
    """测试logspace对数等分
    Test logspace function"""

    def setUp(self):
        paddle.disable_static()

    def test_logspace_base10(self):
        """base=10的logspace / Logspace with base 10"""
        out = paddle.logspace(0, 3, 4, base=10)
        expected = np.array([1.0, 10.0, 100.0, 1000.0], dtype='float32')
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4)

    def test_logspace_base2(self):
        """base=2的logspace / Logspace with base 2"""
        out = paddle.logspace(0, 4, 5, base=2)
        expected = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype='float32')
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4)

    def test_logspace_float64(self):
        """float64的logspace / Logspace with float64"""
        out = paddle.logspace(0, 2, 3, base=10, dtype='float64')
        expected = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-10)


class TestArange(unittest.TestCase):
    """测试arange等差序列
    Test arange function"""

    def setUp(self):
        paddle.disable_static()

    def test_arange_basic(self):
        """基本arange / Basic arange"""
        out = paddle.arange(5)
        np.testing.assert_array_equal(out.numpy(), np.arange(5))

    def test_arange_start_end(self):
        """指定start和end / Arange with start and end"""
        out = paddle.arange(2, 8)
        np.testing.assert_array_equal(out.numpy(), np.arange(2, 8))

    def test_arange_with_step(self):
        """指定step / Arange with step"""
        out = paddle.arange(0, 10, 2, dtype='int32')
        np.testing.assert_array_equal(
            out.numpy(), np.arange(0, 10, 2, dtype='int32')
        )

    def test_arange_float_step(self):
        """浮点步长 / Arange with float step"""
        out = paddle.arange(0.0, 1.0, 0.2, dtype='float32')
        expected = np.arange(0.0, 1.0, 0.2).astype('float32')
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_arange_negative_step(self):
        """负步长 / Arange with negative step"""
        out = paddle.arange(5, 0, -1, dtype='int32')
        np.testing.assert_array_equal(
            out.numpy(), np.arange(5, 0, -1, dtype='int32')
        )


class TestFullAndZerosOnes(unittest.TestCase):
    """测试full, zeros, ones创建
    Test full, zeros, ones creation"""

    def setUp(self):
        paddle.disable_static()

    def test_full_basic(self):
        """基本full / Basic full"""
        out = paddle.full([2, 3], 5.0)
        np.testing.assert_allclose(out.numpy(), np.full([2, 3], 5.0))

    def test_full_int(self):
        """int类型full / Full with int dtype"""
        out = paddle.full([3], 7, dtype='int32')
        np.testing.assert_array_equal(
            out.numpy(), np.array([7, 7, 7], dtype='int32')
        )

    def test_zeros_various_shapes(self):
        """各种形状的zeros / Zeros with various shapes"""
        for shape in [[1], [2, 3], [2, 3, 4]]:
            out = paddle.zeros(shape, dtype='float32')
            self.assertEqual(list(out.shape), shape)
            np.testing.assert_allclose(
                out.numpy(), np.zeros(shape, dtype='float32')
            )

    def test_ones_various_shapes(self):
        """各种形状的ones / Ones with various shapes"""
        for shape in [[1], [2, 3], [2, 3, 4]]:
            out = paddle.ones(shape, dtype='float32')
            self.assertEqual(list(out.shape), shape)
            np.testing.assert_allclose(
                out.numpy(), np.ones(shape, dtype='float32')
            )


class TestMeshgrid(unittest.TestCase):
    """测试meshgrid网格生成
    Test meshgrid function"""

    def setUp(self):
        paddle.disable_static()

    def test_meshgrid_2d(self):
        """2D网格 / 2D meshgrid"""
        x = paddle.arange(3, dtype='float32')
        y = paddle.arange(4, dtype='float32')
        grid_x, grid_y = paddle.meshgrid(x, y)
        self.assertEqual(list(grid_x.shape), [3, 4])
        self.assertEqual(list(grid_y.shape), [3, 4])

    def test_meshgrid_3d(self):
        """3D网格 / 3D meshgrid"""
        x = paddle.arange(2, dtype='float32')
        y = paddle.arange(3, dtype='float32')
        z = paddle.arange(4, dtype='float32')
        gx, gy, gz = paddle.meshgrid(x, y, z)
        self.assertEqual(list(gx.shape), [2, 3, 4])


if __name__ == '__main__':
    unittest.main()
