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
位操作和整数运算测试 / Bitwise and Integer Operations Tests

测试目标 / Test Target:
  paddle.tensor 位操作和整数运算

覆盖的模块 / Covered Modules:
  - paddle.bitwise_and/or/xor/not: 位运算
  - paddle.bitwise_left_shift/right_shift: 位移
  - int64张量操作

作用 / Purpose:
  补充位操作和整数运算API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestBitwiseOps(unittest.TestCase):
    """测试位运算 / Test bitwise operations"""

    def test_bitwise_and(self):
        """测试按位与 / Test bitwise AND"""
        x = paddle.to_tensor([0b1010, 0b1100, 0b1111], dtype='int32')
        y = paddle.to_tensor([0b1111, 0b1010, 0b0101], dtype='int32')
        result = paddle.bitwise_and(x, y)
        np.testing.assert_array_equal(
            result.numpy(), [0b1010 & 0b1111, 0b1100 & 0b1010, 0b1111 & 0b0101]
        )

    def test_bitwise_or(self):
        """测试按位或 / Test bitwise OR"""
        x = paddle.to_tensor([0b1010, 0b1100], dtype='int32')
        y = paddle.to_tensor([0b0101, 0b0011], dtype='int32')
        result = paddle.bitwise_or(x, y)
        np.testing.assert_array_equal(
            result.numpy(), [0b1010 | 0b0101, 0b1100 | 0b0011]
        )

    def test_bitwise_xor(self):
        """测试按位异或 / Test bitwise XOR"""
        x = paddle.to_tensor([0b1010, 0b1111], dtype='int32')
        y = paddle.to_tensor([0b1111, 0b1111], dtype='int32')
        result = paddle.bitwise_xor(x, y)
        np.testing.assert_array_equal(
            result.numpy(), [0b1010 ^ 0b1111, 0b1111 ^ 0b1111]
        )

    def test_bitwise_not(self):
        """测试按位非 / Test bitwise NOT"""
        x = paddle.to_tensor([0, 1, -1], dtype='int32')
        result = paddle.bitwise_not(x)
        np.testing.assert_array_equal(
            result.numpy(), ~np.array([0, 1, -1], dtype=np.int32)
        )

    def test_bitwise_left_shift(self):
        """测试左移 / Test left shift"""
        x = paddle.to_tensor([1, 2, 4], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        result = paddle.bitwise_left_shift(x, y)
        np.testing.assert_array_equal(result.numpy(), [2, 8, 32])

    def test_bitwise_right_shift(self):
        """测试右移 / Test right shift"""
        x = paddle.to_tensor([8, 16, 32], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        result = paddle.bitwise_right_shift(x, y)
        np.testing.assert_array_equal(result.numpy(), [4, 4, 4])


class TestIntegerOps(unittest.TestCase):
    """测试整数运算 / Test integer operations"""

    def test_int_add(self):
        """测试整数加法 / Test integer addition"""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        y = paddle.to_tensor([4, 5, 6], dtype='int64')
        result = x + y
        np.testing.assert_array_equal(result.numpy(), [5, 7, 9])

    def test_int_multiply(self):
        """测试整数乘法 / Test integer multiplication"""
        x = paddle.to_tensor([2, 3, 4], dtype='int32')
        y = paddle.to_tensor([3, 4, 5], dtype='int32')
        result = x * y
        np.testing.assert_array_equal(result.numpy(), [6, 12, 20])

    def test_int_floor_divide(self):
        """测试整数整除 / Test integer floor divide"""
        x = paddle.to_tensor([10, 11, 12], dtype='int32')
        y = paddle.to_tensor([3, 4, 5], dtype='int32')
        result = x // y
        np.testing.assert_array_equal(result.numpy(), [3, 2, 2])

    def test_int_mod(self):
        """测试整数取模 / Test integer modulo"""
        x = paddle.to_tensor([10, 11, 12], dtype='int32')
        y = paddle.to_tensor([3, 4, 5], dtype='int32')
        result = x % y
        np.testing.assert_array_equal(result.numpy(), [1, 3, 2])


class TestTypeConversion(unittest.TestCase):
    """测试类型转换 / Test type conversion"""

    def test_float_to_int(self):
        """测试浮点转整数 / Test float to int conversion"""
        x = paddle.to_tensor([1.7, 2.9, 3.1], dtype='float32')
        result = paddle.cast(x, 'int32')
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3])

    def test_int_to_float(self):
        """测试整数转浮点 / Test int to float conversion"""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        result = paddle.cast(x, 'float32')
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0])

    def test_bool_to_int(self):
        """测试布尔转整数 / Test bool to int conversion"""
        x = paddle.to_tensor([True, False, True])
        result = paddle.cast(x, 'int32')
        np.testing.assert_array_equal(result.numpy(), [1, 0, 1])

    def test_int32_to_int64(self):
        """测试int32转int64 / Test int32 to int64"""
        x = paddle.to_tensor([100, 200, 300], dtype='int32')
        result = paddle.cast(x, 'int64')
        self.assertEqual(result.dtype, paddle.int64)


if __name__ == '__main__':
    unittest.main()
