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

# [AUTO-GENERATED] Tests for paddle/tensor/attribute.py (coverage: 90.0% -> higher)
# Target file: python/paddle/tensor/attribute.py
# Functions: rank, shape, is_complex, is_floating_point, is_integer, imag

import unittest

import numpy as np

import paddle


class TestRank(unittest.TestCase):
    """测试 rank 功能 / Test rank functionality."""

    def test_rank_3d(self):
        """测试 3D 张量 rank / Test rank of 3D tensor."""
        x = paddle.rand([3, 100, 100])
        out = paddle.rank(x)
        np.testing.assert_equal(out.numpy(), 3)

    def test_rank_0d(self):
        """测试标量 rank / Test rank of scalar tensor."""
        x = paddle.to_tensor(5.0)
        out = paddle.rank(x)
        np.testing.assert_equal(out.numpy(), 0)

    def test_rank_1d(self):
        """测试 1D 张量 rank / Test rank of 1D tensor."""
        x = paddle.randn([10])
        out = paddle.rank(x)
        np.testing.assert_equal(out.numpy(), 1)

    def test_rank_4d(self):
        """测试 4D 张量 rank / Test rank of 4D tensor."""
        x = paddle.randn([2, 3, 4, 5])
        out = paddle.rank(x)
        np.testing.assert_equal(out.numpy(), 4)


class TestShape(unittest.TestCase):
    """测试 shape 功能 / Test shape functionality."""

    def test_shape_2d(self):
        """测试 2D 张量 shape / Test shape of 2D tensor."""
        x = paddle.randn([3, 5])
        out = paddle.shape(x)
        np.testing.assert_array_equal(out.numpy(), [3, 5])

    def test_shape_3d(self):
        """测试 3D 张量 shape / Test shape of 3D tensor."""
        x = paddle.randn([2, 3, 4])
        out = paddle.shape(x)
        np.testing.assert_array_equal(out.numpy(), [2, 3, 4])

    def test_shape_0d(self):
        """测试标量 shape / Test shape of scalar tensor."""
        x = paddle.to_tensor(5.0)
        out = paddle.shape(x)
        np.testing.assert_array_equal(out.numpy(), [])


class TestIsComplex(unittest.TestCase):
    """测试 is_complex 功能 / Test is_complex functionality."""

    def test_is_complex_true64(self):
        """测试 complex64 返回 True / Test complex64 returns True."""
        x = paddle.to_tensor([1 + 2j])
        self.assertTrue(paddle.is_complex(x))

    def test_is_complex_true128(self):
        """测试 complex128 返回 True / Test complex128 returns True."""
        x = paddle.to_tensor([1 + 2j], dtype='complex128')
        self.assertTrue(paddle.is_complex(x))

    def test_is_complex_false_float(self):
        """测试 float 返回 False / Test float returns False."""
        x = paddle.to_tensor([1.1, 2.2])
        self.assertFalse(paddle.is_complex(x))

    def test_is_complex_false_int(self):
        """测试 int 返回 False / Test int returns False."""
        x = paddle.to_tensor([1, 2, 3])
        self.assertFalse(paddle.is_complex(x))

    def test_is_complex_false_bool(self):
        """测试 bool 返回 False / Test bool returns False."""
        x = paddle.to_tensor([True, False])
        self.assertFalse(paddle.is_complex(x))

    def test_is_complex_type_error(self):
        """测试非张量输入报错 / Test non-tensor input raises error."""
        with self.assertRaises(TypeError):
            paddle.is_complex([1 + 2j])

    def test_is_complex_alias(self):
        """测试 input 别名 / Test is_complex with input alias."""
        x = paddle.to_tensor([1 + 2j])
        self.assertTrue(paddle.is_complex(input=x))


class TestIsFloatingPoint(unittest.TestCase):
    """测试 is_floating_point 功能 / Test is_floating_point functionality."""

    def test_is_floating_point_true32(self):
        """测试 float32 返回 True / Test float32 returns True."""
        x = paddle.arange(1.0, 5.0, dtype='float32')
        self.assertTrue(paddle.is_floating_point(x))

    def test_is_floating_point_true64(self):
        """测试 float64 返回 True / Test float64 returns True."""
        x = paddle.arange(1.0, 5.0, dtype='float64')
        self.assertTrue(paddle.is_floating_point(x))

    def test_is_floating_point_true16(self):
        """测试 float16 返回 True / Test float16 returns True."""
        x = paddle.arange(1.0, 5.0, dtype='float16')
        self.assertTrue(paddle.is_floating_point(x))

    def test_is_floating_point_true_bf16(self):
        """测试 bfloat16 返回 True / Test bfloat16 returns True."""
        x = paddle.arange(1.0, 5.0, dtype='bfloat16')
        self.assertTrue(paddle.is_floating_point(x))

    def test_is_floating_point_false_int(self):
        """测试 int 返回 False / Test int returns False."""
        x = paddle.arange(1, 5, dtype='int32')
        self.assertFalse(paddle.is_floating_point(x))

    def test_is_floating_point_false_int64(self):
        """测试 int64 返回 False / Test int64 returns False."""
        x = paddle.arange(1, 5, dtype='int64')
        self.assertFalse(paddle.is_floating_point(x))

    def test_is_floating_point_type_error(self):
        """测试非张量输入报错 / Test non-tensor input raises error."""
        with self.assertRaises(TypeError):
            paddle.is_floating_point([1.0])

    def test_is_floating_point_alias(self):
        """测试 input 别名 / Test is_floating_point with input alias."""
        x = paddle.arange(1.0, 5.0, dtype='float32')
        self.assertTrue(paddle.is_floating_point(input=x))


class TestIsInteger(unittest.TestCase):
    """测试 is_integer 功能 / Test is_integer functionality."""

    def test_is_integer_true_int32(self):
        """测试 int32 返回 True / Test int32 returns True."""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.assertTrue(paddle.is_integer(x))

    def test_is_integer_true_int64(self):
        """测试 int64 返回 True / Test int64 returns True."""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        self.assertTrue(paddle.is_integer(x))

    def test_is_integer_false_float(self):
        """测试 float 返回 False / Test float returns False."""
        x = paddle.to_tensor([1.0, 2.0])
        self.assertFalse(paddle.is_integer(x))

    def test_is_integer_false_complex(self):
        """测试 complex 返回 False / Test complex returns False."""
        x = paddle.to_tensor([1 + 2j])
        self.assertFalse(paddle.is_integer(x))

    def test_is_integer_false_bool(self):
        """测试 bool 返回 False / Test bool returns False."""
        x = paddle.to_tensor([True, False])
        self.assertFalse(paddle.is_integer(x))

    def test_is_integer_type_error(self):
        """测试非张量输入报错 / Test non-tensor input raises error."""
        with self.assertRaises(TypeError):
            paddle.is_integer([1, 2, 3])


class TestImag(unittest.TestCase):
    """测试 imag 功能 / Test imag functionality."""

    def test_imag_basic(self):
        """测试 imag 基本功能 / Test basic imag."""
        x = paddle.to_tensor([1 + 6j, 2 + 5j, 3 + 4j])
        out = paddle.imag(x)
        np.testing.assert_array_almost_equal(out.numpy(), [6.0, 5.0, 4.0])

    def test_imag_2d(self):
        """测试 2D 张量 imag / Test imag of 2D tensor."""
        x = paddle.to_tensor([[1 + 6j, 2 + 5j], [4 + 3j, 5 + 2j]])
        out = paddle.imag(x)
        np.testing.assert_array_almost_equal(
            out.numpy(), [[6.0, 5.0], [3.0, 2.0]]
        )

    def test_imag_tensor_method(self):
        """测试张量 .imag() 方法 / Test tensor .imag() method."""
        x = paddle.to_tensor([1 + 6j, 2 + 5j])
        out = x.imag()
        np.testing.assert_array_almost_equal(out.numpy(), [6.0, 5.0])

    def test_imag_complex128(self):
        """测试 complex128 imag / Test imag of complex128 tensor."""
        x = paddle.to_tensor([1 + 6j], dtype='complex128')
        out = paddle.imag(x)
        np.testing.assert_array_almost_equal(out.numpy(), [6.0])
        self.assertEqual(out.dtype, paddle.float64)


if __name__ == '__main__':
    unittest.main()
