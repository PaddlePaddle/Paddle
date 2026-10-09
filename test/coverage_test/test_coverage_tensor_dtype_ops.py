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

"""
数据类型和张量属性测试 / Data Type and Tensor Attribute Tests

测试目标 / Test Target:
  paddle.tensor 数据类型和属性

覆盖的模块 / Covered Modules:
  - 各种数据类型创建和转换
  - 张量属性: shape, dtype, device, ndim
  - paddle.Tensor方法

作用 / Purpose:
  补充张量属性和数据类型API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestTensorDtypes(unittest.TestCase):
    """测试张量数据类型 / Test tensor data types"""

    def test_float32(self):
        """测试float32张量 / Test float32 tensor"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        self.assertEqual(x.dtype, paddle.float32)

    def test_float64(self):
        """测试float64张量 / Test float64 tensor"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float64')
        self.assertEqual(x.dtype, paddle.float64)

    def test_float16(self):
        """测试float16张量 / Test float16 tensor"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float16')
        self.assertEqual(x.dtype, paddle.float16)

    def test_int32(self):
        """测试int32张量 / Test int32 tensor"""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.assertEqual(x.dtype, paddle.int32)

    def test_int64(self):
        """测试int64张量 / Test int64 tensor"""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        self.assertEqual(x.dtype, paddle.int64)

    def test_bool(self):
        """测试bool张量 / Test bool tensor"""
        x = paddle.to_tensor([True, False, True])
        self.assertEqual(x.dtype, paddle.bool)

    def test_complex64(self):
        """测试complex64张量 / Test complex64 tensor"""
        x = paddle.to_tensor([1 + 2j, 3 + 4j], dtype='complex64')
        self.assertEqual(x.dtype, paddle.complex64)

    def test_complex128(self):
        """测试complex128张量 / Test complex128 tensor"""
        x = paddle.to_tensor([1 + 2j, 3 + 4j], dtype='complex128')
        self.assertEqual(x.dtype, paddle.complex128)


class TestTensorProperties(unittest.TestCase):
    """测试张量属性 / Test tensor properties"""

    def test_shape(self):
        """测试shape属性 / Test shape property"""
        x = paddle.randn([2, 3, 4])
        self.assertEqual(x.shape, [2, 3, 4])

    def test_ndim(self):
        """测试ndim属性 / Test ndim property"""
        x = paddle.randn([2, 3, 4])
        self.assertEqual(x.ndim, 3)

    def test_numel(self):
        """测试元素总数 / Test total number of elements"""
        x = paddle.randn([2, 3, 4])
        self.assertEqual(x.numel(), 24)

    def test_size(self):
        """测试size方法 / Test size method"""
        x = paddle.randn([2, 3])
        self.assertEqual(x.size, 6)

    def test_is_floating_point(self):
        """测试浮点类型检查 / Test floating point type check"""
        float_x = paddle.randn([3])
        int_x = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.assertTrue(float_x.is_floating_point())
        self.assertFalse(int_x.is_floating_point())

    def test_stop_gradient(self):
        """测试stop_gradient属性 / Test stop_gradient property"""
        x = paddle.randn([3])
        self.assertTrue(x.stop_gradient)
        x.stop_gradient = False
        self.assertFalse(x.stop_gradient)

    def test_tensor_place(self):
        """测试张量设备 / Test tensor device"""
        x = paddle.randn([3])
        self.assertIsNotNone(x.place)


class TestTensorOperators(unittest.TestCase):
    """测试张量运算符 / Test tensor operators"""

    def test_add_operator(self):
        """测试加法运算符 / Test addition operator"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        result = x + y
        np.testing.assert_allclose(result.numpy(), [5.0, 7.0, 9.0])

    def test_subtract_operator(self):
        """测试减法运算符 / Test subtraction operator"""
        x = paddle.to_tensor([4.0, 5.0, 6.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        result = x - y
        np.testing.assert_allclose(result.numpy(), [3.0, 3.0, 3.0])

    def test_multiply_operator(self):
        """测试乘法运算符 / Test multiplication operator"""
        x = paddle.to_tensor([2.0, 3.0, 4.0])
        y = paddle.to_tensor([3.0, 4.0, 5.0])
        result = x * y
        np.testing.assert_allclose(result.numpy(), [6.0, 12.0, 20.0])

    def test_divide_operator(self):
        """测试除法运算符 / Test division operator"""
        x = paddle.to_tensor([6.0, 8.0, 9.0])
        y = paddle.to_tensor([2.0, 4.0, 3.0])
        result = x / y
        np.testing.assert_allclose(result.numpy(), [3.0, 2.0, 3.0])

    def test_neg_operator(self):
        """测试负号运算符 / Test negation operator"""
        x = paddle.to_tensor([1.0, -2.0, 3.0])
        result = -x
        np.testing.assert_allclose(result.numpy(), [-1.0, 2.0, -3.0])

    def test_comparison_operators(self):
        """测试比较运算符 / Test comparison operators"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([2.0, 2.0, 2.0])
        np.testing.assert_array_equal((x < y).numpy(), [True, False, False])
        np.testing.assert_array_equal((x <= y).numpy(), [True, True, False])
        np.testing.assert_array_equal((x > y).numpy(), [False, False, True])
        np.testing.assert_array_equal((x >= y).numpy(), [False, True, True])
        np.testing.assert_array_equal((x == y).numpy(), [False, True, False])
        np.testing.assert_array_equal((x != y).numpy(), [True, False, True])


if __name__ == '__main__':
    unittest.main()
