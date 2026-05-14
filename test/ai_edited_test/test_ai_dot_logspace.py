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

# [AUTO-GENERATED] Tests for phi/kernels/cpu/dot_kernel.cc and phi/kernels/cpu/logspace_kernel.cc
# dot_kernel.cc: CPU dot product kernel (1D/2D, float/double/int/int64/complex)
# logspace_kernel.cc: CPU logspace kernel (geometric spacing via base^exponent)

import unittest

import numpy as np

import paddle


class TestDotKernel(unittest.TestCase):
    """Test suite for paddle.dot CPU kernel.

    测试 paddle.dot 的 CPU 内核，涵盖 1D/2D 张量、不同数据类型、空张量等场景。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_dot_1d_basic(self):
        """Test basic 1D dot product.

        测试基本的 1D 向量点积运算。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.dot(x, y)
        expected = np.array(32.0, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_1d_negative(self):
        """Test 1D dot product with negative values.

        测试包含负值的 1D 向量点积。
        """
        x = paddle.to_tensor([-1.0, 2.0, -3.0])
        y = paddle.to_tensor([4.0, -5.0, 6.0])
        result = paddle.dot(x, y)
        expected = np.array(-32.0, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_2d(self):
        """Test 2D dot product (batched dot along last axis).

        测试 2D 张量的点积运算（沿最后一个轴批量计算）。
        """
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        result = paddle.dot(x, y)
        # dot along last axis: [1*5+2*6, 3*7+4*8] = [17, 53]
        expected = np.array([17.0, 53.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_2d_batch(self):
        """Test 2D dot product with multiple rows.

        测试多行 2D 张量的点积运算。
        """
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = paddle.to_tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        result = paddle.dot(x, y)
        # [1+2+3, 8+10+12] = [6, 30]
        expected = np.array([6.0, 30.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_float64(self):
        """Test dot product with float64 dtype.

        测试 float64 数据类型的点积运算。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        y = paddle.to_tensor([4.0, 5.0, 6.0], dtype='float64')
        result = paddle.dot(x, y)
        expected = np.array(32.0, dtype=np.float64)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10)

    def test_dot_int32(self):
        """Test dot product with int32 dtype.

        测试 int32 数据类型的点积运算。
        """
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        y = paddle.to_tensor([4, 5, 6], dtype='int32')
        result = paddle.dot(x, y)
        expected = np.array(32, dtype=np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_dot_int64(self):
        """Test dot product with int64 dtype.

        测试 int64 数据类型的点积运算。
        """
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        y = paddle.to_tensor([4, 5, 6], dtype='int64')
        result = paddle.dot(x, y)
        expected = np.array(32, dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_dot_complex64(self):
        """Test dot product with complex64 dtype.

        测试 complex64 复数数据类型的点积运算。
        """
        x = paddle.to_tensor([1 + 2j, 3 + 4j], dtype='complex64')
        y = paddle.to_tensor([5 + 6j, 7 + 8j], dtype='complex64')
        result = paddle.dot(x, y)
        # (1+2j)(5+6j) + (3+4j)(7+8j)
        # = (-7+16j) + (-11+52j) = -18+68j
        expected = np.array(-18 + 68j, dtype=np.complex64)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_dot_empty(self):
        """Test dot product with empty tensors.

        测试空张量的点积运算（应返回 0）。
        """
        x = paddle.to_tensor([], dtype='float32')
        y = paddle.to_tensor([], dtype='float32')
        result = paddle.dot(x, y)
        expected = np.array(0.0, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_single_element(self):
        """Test dot product with single-element tensors.

        测试单元素张量的点积运算。
        """
        x = paddle.to_tensor([5.0])
        y = paddle.to_tensor([3.0])
        result = paddle.dot(x, y)
        expected = np.array(15.0, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_dot_zeros(self):
        """Test dot product with zeros.

        测试全零向量的点积运算。
        """
        x = paddle.to_tensor([0.0, 0.0, 0.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.dot(x, y)
        expected = np.array(0.0, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


class TestLogspaceKernel(unittest.TestCase):
    """Test suite for paddle.logspace CPU kernel.

    测试 paddle.logspace 的 CPU 内核，涵盖不同底数、数据类型、边界情况等场景。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_logspace_base10(self):
        """Test logspace with default base 10.

        测试默认底数为 10 的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=10.0)
        expected = np.array(
            [1.0, 10**0.5, 10.0, 10**1.5, 100.0], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_base2(self):
        """Test logspace with base 2.

        测试底数为 2 的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=2.0)
        expected = np.array(
            [2**0, 2**0.5, 2**1, 2**1.5, 2**2], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_base_e(self):
        """Test logspace with base e.

        测试底数为 e 的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=np.e)
        expected = np.array(
            [np.e**0, np.e**0.5, np.e**1, np.e**1.5, np.e**2], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_num1(self):
        """Test logspace with num=1 (single element).

        测试 num=1 时仅返回 base^start。
        """
        result = paddle.logspace(2, 5, 1, base=10.0)
        expected = np.array([100.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_num2(self):
        """Test logspace with num=2 (start and stop only).

        测试 num=2 时仅返回 base^start 和 base^stop。
        """
        result = paddle.logspace(0, 3, 2, base=10.0)
        expected = np.array([1.0, 1000.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_negative_exponent(self):
        """Test logspace with negative exponents.

        测试负指数的对数等间距序列。
        """
        result = paddle.logspace(-2, 2, 5, base=10.0)
        expected = np.array(
            [10 ** (-2), 10 ** (-1), 10**0, 10**1, 10**2], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_float64(self):
        """Test logspace with float64 dtype.

        测试 float64 数据类型的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=10.0, dtype='float64')
        expected = np.array(
            [1.0, 10**0.5, 10.0, 10**1.5, 100.0], dtype=np.float64
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10)

    def test_logspace_int32(self):
        """Test logspace with int32 dtype (truncated values).

        测试 int32 数据类型，验证结果被截断为整数。
        """
        result = paddle.logspace(0, 2, 5, base=10.0, dtype='int32')
        expected = np.array([1, 3, 10, 31, 100], dtype=np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logspace_int64(self):
        """Test logspace with int64 dtype.

        测试 int64 数据类型的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=10.0, dtype='int64')
        expected = np.array([1, 3, 10, 31, 100], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logspace_tensor_start_stop(self):
        """Test logspace with Tensor inputs for start/stop.

        测试使用 Tensor 作为 start 和 stop 参数。
        """
        start = paddle.to_tensor(0.0)
        stop = paddle.to_tensor(2.0)
        result = paddle.logspace(start, stop, 5, base=10.0)
        expected = np.array(
            [1.0, 10**0.5, 10.0, 10**1.5, 100.0], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_symmetry(self):
        """Test logspace numerical symmetry (half from start, half from stop).

        测试 logspace 的数值对称性：前半部分从 start 计算，后半部分从 stop 计算，
        确保中间值的一致性。
        """
        result = paddle.logspace(0, 4, 9, base=10.0)
        # The kernel computes first half from start, second half from stop
        # Middle element should be exactly 10^2 = 100
        self.assertAlmostEqual(result.numpy()[4], 100.0, places=4)

    def test_logspace_base3(self):
        """Test logspace with base 3.

        测试底数为 3 的对数等间距序列。
        """
        result = paddle.logspace(0, 2, 5, base=3.0)
        expected = np.array(
            [3**0, 3**0.5, 3**1, 3**1.5, 3**2], dtype=np.float32
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
