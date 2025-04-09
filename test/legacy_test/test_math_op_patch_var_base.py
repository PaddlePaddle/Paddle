#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import unittest

import numpy as np
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base


class TestMathOpPatchesVarBase(unittest.TestCase):
    def setUp(self):
        self.shape = [10, 1024]
        self.dtype = np.float32

    def test_add(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F2_F4(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F2_F8(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F4_F8(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F2_BF(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F4_BF(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_type_promotion_add_F8_BF(self):
        a_np = np.random.random(self.shape).astype(np.float64)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a + b
            res_t = b + a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_sub(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a - b
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)

    def test_type_promotion_sub_F2_F4(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_type_promotion_sub_F2_F8(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_type_promotion_sub_F4_F8(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_type_promotion_sub_F2_BF(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_type_promotion_sub_F4_BF(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_type_promotion_sub_F8_BF(self):
        a_np = np.random.random(self.shape).astype(np.float64)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a - b
            res_t = b - a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)
            np.testing.assert_array_equal(res_t.numpy(), b_np - a_np)

    def test_mul(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a * b
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F2_F4(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F2_F8(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F4_F8(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float64)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F2_BF(self):
        a_np = np.random.random(self.shape).astype(np.float16)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F4_BF(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float32, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_type_promotion_mul_F8_BF(self):
        a_np = np.random.random(self.shape).astype(np.float64)
        b_np = np.random.random(self.shape).astype(np.float32)
        b_np = convert_uint16_to_float(convert_float_to_uint16(b_np))
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np).astype('bfloat16')
            res = a * b
            res_t = b * a
            np.testing.assert_equal(paddle.float64, res.dtype)
            np.testing.assert_equal(res_t.dtype, res.dtype)
            np.testing.assert_array_equal(res_t.numpy(), res.numpy())
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_div(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a / b
            # NOTE: Not sure why array_equal fails on windows, allclose is acceptable
            np.testing.assert_allclose(res.numpy(), a_np / b_np, rtol=1e-05)

    def test_add_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b)

    def test_add_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = b + a
            np.testing.assert_array_equal(res.numpy(), b + a_np)

    def test_sub_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = a - b
            np.testing.assert_array_equal(res.numpy(), a_np - b)

    def test_sub_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = b - a
            np.testing.assert_array_equal(res.numpy(), b - a_np)

    def test_mul_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = a * b
            np.testing.assert_array_equal(res.numpy(), a_np * b)

    # div_scalar, not equal
    def test_div_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = 0.1
            res = a / b
            np.testing.assert_allclose(res.numpy(), a_np / b, rtol=1e-05)

    # pow of float type, not equal
    def test_pow(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a**b
            np.testing.assert_allclose(res.numpy(), a_np**b_np, rtol=1e-05)

    def test_floor_div(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a // b
            np.testing.assert_array_equal(res.numpy(), a_np // b_np)

    def test_mod(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a % b
            np.testing.assert_array_equal(res.numpy(), a_np % b_np)

    # for bitwise and/or/xor/not
    def test_bitwise(self):
        paddle.disable_static()

        x_np = np.random.randint(-100, 100, [2, 3, 5])
        y_np = np.random.randint(-100, 100, [2, 3, 5])
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)

        out_np = x_np & y_np
        out = x & y
        np.testing.assert_array_equal(out.numpy(), out_np)

        out_np = x_np | y_np
        out = x | y
        np.testing.assert_array_equal(out.numpy(), out_np)

        out_np = x_np ^ y_np
        out = x ^ y
        np.testing.assert_array_equal(out.numpy(), out_np)

        out_np = ~x_np
        out = ~x
        np.testing.assert_array_equal(out.numpy(), out_np)

    # for logical compare
    def test_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            c = paddle.to_tensor(c_np)
            res1 = a == b
            res2 = a == c
            np.testing.assert_array_equal(res1.numpy(), a_np == b_np)
            np.testing.assert_array_equal(res2.numpy(), a_np == c_np)

    def test_not_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            c = paddle.to_tensor(c_np)
            res1 = a != b
            res2 = a != c
            np.testing.assert_array_equal(res1.numpy(), a_np != b_np)
            np.testing.assert_array_equal(res2.numpy(), a_np != c_np)

    def test_less_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a < b
            np.testing.assert_array_equal(res.numpy(), a_np < b_np)

    def test_less(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res_tensor = a.less(b)
            res_paddle = paddle.less(a, b)
            np.testing.assert_array_equal(res_tensor.numpy(), a_np < b_np)
            np.testing.assert_array_equal(res_paddle.numpy(), a_np < b_np)

    def test_less_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a <= b
            np.testing.assert_array_equal(res.numpy(), a_np <= b_np)

    def test_greater_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a > b
            np.testing.assert_array_equal(res.numpy(), a_np > b_np)

    def test_greater_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a >= b
            np.testing.assert_array_equal(res.numpy(), a_np >= b_np)

    def test_neg(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            res = -a
            np.testing.assert_array_equal(res.numpy(), -a_np)

    def test_abs(self):
        # test for real number
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            res = abs(a)
            np.testing.assert_array_equal(res.numpy(), np.abs(a_np))

    def test_abs_complex(self):
        # test for complex number
        a_np = np.random.uniform(-1, 1, self.shape).astype(
            self.dtype
        ) + 1j * np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            res = abs(a)
            np.testing.assert_allclose(
                res.numpy(), np.abs(a_np), rtol=2e-7, atol=0.0
            )

    def test_float_int_long(self):
        with base.dygraph.guard():
            a = paddle.to_tensor(np.array([100.1]))
            self.assertTrue(float(a) == 100.1)
            self.assertTrue(int(a) == 100)
            self.assertTrue(int(a) == 100)

        a = paddle.to_tensor(1000000.0, dtype='bfloat16')
        self.assertTrue(float(a) == 999424.0)
        self.assertTrue(int(a) == 999424)
        self.assertTrue(int(a) == 999424)

    def test_complex(self):
        with base.dygraph.guard():
            a = paddle.to_tensor(np.array([100.1 + 99.9j]))
            self.assertTrue(complex(a) == (100.1 + 99.9j))

        a = paddle.to_tensor(1000000.0, dtype='bfloat16')
        self.assertTrue(complex(a) == (999424 + 0j))

    def test_len(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            self.assertTrue(len(a) == 10)

    def test_index(self):
        with base.dygraph.guard():
            var1 = paddle.to_tensor(np.array([2]))
            i_tmp = 0
            for i in range(var1):
                self.assertTrue(i == i_tmp)
                i_tmp = i_tmp + 1
            list1 = [1, 2, 3, 4, 5]
            self.assertTrue(list1[var1] == 3)
            str1 = "just test"
            self.assertTrue(str1[var1] == 's')

        var1 = paddle.to_tensor(2.0, dtype='bfloat16')
        i_tmp = 0
        for i in range(var1):
            self.assertTrue(i == i_tmp)
            i_tmp = i_tmp + 1
        list1 = [1, 2, 3, 4, 5]
        self.assertTrue(list1[var1] == 3)
        str1 = "just test"
        self.assertTrue(str1[var1] == 's')

    def test_np_left_mul(self):
        with base.dygraph.guard():
            t = np.sqrt(2.0 * np.pi)
            x = paddle.ones((2, 2), dtype="float32")
            y = t * x

            np.testing.assert_allclose(
                y.numpy(),
                t * np.ones((2, 2), dtype='float32'),
                rtol=1e-05,
                atol=0.0,
            )

    def test_add_different_dtype(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float16)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_floordiv_different_dtype(self):
        a_np = np.full(self.shape, 10, np.float32)
        b_np = np.full(self.shape, 2, np.float16)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a // b
            np.testing.assert_array_equal(res.numpy(), a_np // b_np)

    def test_astype(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            res1 = a.astype(np.float16)
            res2 = a.astype('float16')
            res3 = a.astype(paddle.float16)
            res4 = a.astype(a.dtype)

            self.assertEqual(res1.dtype, res2.dtype)
            self.assertEqual(res1.dtype, res3.dtype)
            self.assertEqual(res4.dtype, a.dtype)
            self.assertEqual(
                a.data_ptr(), res4.data_ptr()
            )  # zero-copy if same dtype

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())
            np.testing.assert_array_equal(res1.numpy(), res3.numpy())

    def test_conpare_op_broadcast(self):
        a_np = np.random.uniform(-1, 1, [10, 1, 10]).astype(self.dtype)
        b_np = np.random.uniform(-1, 1, [1, 1, 10]).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)

            self.assertEqual((a != b).dtype, paddle.bool)
            np.testing.assert_array_equal((a != b).numpy(), a_np != b_np)

    def test_tensor_patch_method(self):
        paddle.disable_static()
        x_np = np.random.uniform(-1, 1, [2, 3]).astype(self.dtype)
        y_np = np.random.uniform(-1, 1, [2, 3]).astype(self.dtype)
        z_np = np.random.uniform(-1, 1, [6, 9]).astype(self.dtype)

        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        z = paddle.to_tensor(z_np)

        a = paddle.to_tensor([[1, 1], [2, 2], [3, 3]])
        b = paddle.to_tensor([[1, 1], [2, 2], [3, 3]])

        # 1. Unary operation for Tensor
        self.assertEqual(x.dim(), 2)
        self.assertEqual(x.ndimension(), 2)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.size, 6)
        self.assertEqual(x.numel(), 6)
        np.testing.assert_array_equal(x.exp().numpy(), paddle.exp(x).numpy())
        np.testing.assert_array_equal(x.tanh().numpy(), paddle.tanh(x).numpy())
        np.testing.assert_array_equal(x.atan().numpy(), paddle.atan(x).numpy())
        np.testing.assert_array_equal(x.abs().numpy(), paddle.abs(x).numpy())
        m = x.abs()
        np.testing.assert_array_equal(m.sqrt().numpy(), paddle.sqrt(m).numpy())
        np.testing.assert_array_equal(
            m.rsqrt().numpy(), paddle.rsqrt(m).numpy()
        )
        np.testing.assert_array_equal(x.ceil().numpy(), paddle.ceil(x).numpy())
        np.testing.assert_array_equal(
            x.floor().numpy(), paddle.floor(x).numpy()
        )
        np.testing.assert_array_equal(x.cos().numpy(), paddle.cos(x).numpy())
        np.testing.assert_array_equal(x.acos().numpy(), paddle.acos(x).numpy())
        np.testing.assert_array_equal(x.asin().numpy(), paddle.asin(x).numpy())
        np.testing.assert_array_equal(x.sin().numpy(), paddle.sin(x).numpy())
        np.testing.assert_array_equal(x.sinh().numpy(), paddle.sinh(x).numpy())
        np.testing.assert_array_equal(x.cosh().numpy(), paddle.cosh(x).numpy())
        np.testing.assert_array_equal(
            x.round().numpy(), paddle.round(x).numpy()
        )
        np.testing.assert_array_equal(
            x.reciprocal().numpy(), paddle.reciprocal(x).numpy()
        )
        np.testing.assert_array_equal(
            x.square().numpy(), paddle.square(x).numpy()
        )
        np.testing.assert_array_equal(x.rank().numpy(), paddle.rank(x).numpy())
        np.testing.assert_array_equal(x[0].t().numpy(), paddle.t(x[0]).numpy())
        np.testing.assert_array_equal(
            x.asinh().numpy(), paddle.asinh(x).numpy()
        )
        # acosh(x) = nan, need to change input
        t_np = np.random.uniform(1, 2, [2, 3]).astype(self.dtype)
        t = paddle.to_tensor(t_np)
        np.testing.assert_array_equal(
            t.acosh().numpy(), paddle.acosh(t).numpy()
        )
        np.testing.assert_array_equal(
            x.atanh().numpy(), paddle.atanh(x).numpy()
        )
        d = paddle.to_tensor(
            [
                [1.2285208, 1.3491015, 1.4899898],
                [1.30058, 1.0688717, 1.4928783],
                [1.0958099, 1.3724753, 1.8926544],
            ]
        )
        d = d.matmul(d.t())
        # ROCM not support cholesky
        if not base.core.is_compiled_with_rocm():
            np.testing.assert_array_equal(
                d.cholesky().numpy(), paddle.cholesky(d).numpy()
            )

        np.testing.assert_array_equal(
            x.is_empty().numpy(), paddle.is_empty(x).numpy()
        )
        np.testing.assert_array_equal(
            x.isfinite().numpy(), paddle.isfinite(x).numpy()
        )
        np.testing.assert_array_equal(
            x.cast('int32').numpy(), paddle.cast(x, 'int32').numpy()
        )
        np.testing.assert_array_equal(
            x.expand([3, 2, 3]).numpy(), paddle.expand(x, [3, 2, 3]).numpy()
        )
        np.testing.assert_array_equal(
            x.tile([2, 2]).numpy(), paddle.tile(x, [2, 2]).numpy()
        )
        np.testing.assert_array_equal(
            x.flatten().numpy(), paddle.flatten(x).numpy()
        )
        index = paddle.to_tensor([0, 1])
        np.testing.assert_array_equal(
            x.gather(index).numpy(), paddle.gather(x, index).numpy()
        )
        index = paddle.to_tensor([[0, 1], [1, 2]])
        np.testing.assert_array_equal(
            x.gather_nd(index).numpy(), paddle.gather_nd(x, index).numpy()
        )
        np.testing.assert_array_equal(
            x.reverse([0, 1]).numpy(), paddle.reverse(x, [0, 1]).numpy()
        )
        np.testing.assert_array_equal(
            a.reshape([3, 2]).numpy(), paddle.reshape(a, [3, 2]).numpy()
        )
        np.testing.assert_array_equal(
            x.slice([0, 1], [0, 0], [1, 2]).numpy(),
            paddle.slice(x, [0, 1], [0, 0], [1, 2]).numpy(),
        )
        np.testing.assert_array_equal(
            x.split(2)[0].numpy(), paddle.split(x, 2)[0].numpy()
        )
        m = paddle.to_tensor(
            np.random.uniform(-1, 1, [1, 6, 1, 1]).astype(self.dtype)
        )
        np.testing.assert_array_equal(
            m.squeeze([]).numpy(), paddle.squeeze(m, []).numpy()
        )
        np.testing.assert_array_equal(
            m.squeeze([1, 2]).numpy(), paddle.squeeze(m, [1, 2]).numpy()
        )
        m = paddle.to_tensor([2, 3, 3, 1, 5, 3], 'float32')
        np.testing.assert_array_equal(
            m.unique()[0].numpy(), paddle.unique(m)[0].numpy()
        )
        np.testing.assert_array_equal(
            m.unique(return_counts=True)[1],
            paddle.unique(m, return_counts=True)[1],
        )
        np.testing.assert_array_equal(x.flip([0]), paddle.flip(x, [0]))
        np.testing.assert_array_equal(x.unbind(0), paddle.unbind(x, 0))
        np.testing.assert_array_equal(x.roll(1), paddle.roll(x, 1))
        np.testing.assert_array_equal(x.cumsum(1), paddle.cumsum(x, 1))
        m = paddle.to_tensor(1)
        np.testing.assert_array_equal(m.increment(), paddle.increment(m))
        m = x.abs()
        np.testing.assert_array_equal(m.log(), paddle.log(m))
        np.testing.assert_array_equal(x.pow(2), paddle.pow(x, 2))
        np.testing.assert_array_equal(x.reciprocal(), paddle.reciprocal(x))

        # 2. Binary operation
        np.testing.assert_array_equal(
            x.divide(y).numpy(), paddle.divide(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.matmul(y, True, False).numpy(),
            paddle.matmul(x, y, True, False).numpy(),
        )
        np.testing.assert_array_equal(
            x.norm(p='fro', axis=[0, 1]).numpy(),
            paddle.norm(x, p='fro', axis=[0, 1]).numpy(),
        )
        np.testing.assert_array_equal(
            x.dist(y).numpy(), paddle.dist(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.cross(y).numpy(), paddle.cross(x, y).numpy()
        )
        m = x.expand([2, 2, 3])
        n = y.expand([2, 2, 3]).transpose([0, 2, 1])
        np.testing.assert_array_equal(
            m.bmm(n).numpy(), paddle.bmm(m, n).numpy()
        )
        np.testing.assert_array_equal(
            x.histogram(5, -1, 1).numpy(), paddle.histogram(x, 5, -1, 1).numpy()
        )
        np.testing.assert_array_equal(
            x.equal(y).numpy(), paddle.equal(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.greater_equal(y).numpy(), paddle.greater_equal(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.greater_than(y).numpy(), paddle.greater_than(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.less_equal(y).numpy(), paddle.less_equal(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.less_than(y).numpy(), paddle.less_than(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.not_equal(y).numpy(), paddle.not_equal(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.equal_all(y).numpy(), paddle.equal_all(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.allclose(y).numpy(), paddle.allclose(x, y).numpy()
        )
        m = x.expand([2, 2, 3])
        np.testing.assert_array_equal(
            x.expand_as(m).numpy(), paddle.expand_as(x, m).numpy()
        )
        index = paddle.to_tensor([2, 1, 0])
        np.testing.assert_array_equal(
            a.scatter(index, b).numpy(), paddle.scatter(a, index, b).numpy()
        )

        # 3. Bool tensor operation
        x = paddle.to_tensor([[True, False], [True, False]])
        y = paddle.to_tensor([[False, False], [False, True]])
        np.testing.assert_array_equal(
            x.logical_and(y).numpy(), paddle.logical_and(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.logical_not(y).numpy(), paddle.logical_not(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.logical_or(y).numpy(), paddle.logical_or(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.logical_xor(y).numpy(), paddle.logical_xor(x, y).numpy()
        )
        np.testing.assert_array_equal(
            x.logical_and(y).numpy(), paddle.logical_and(x, y).numpy()
        )
        a = paddle.to_tensor([[1, 2], [3, 4]])
        b = paddle.to_tensor([[4, 3], [2, 1]])
        np.testing.assert_array_equal(
            x.where(a, b).numpy(), paddle.where(x, a, b).numpy()
        )

        x_np = np.random.randn(3, 6, 9, 7)
        x = paddle.to_tensor(x_np)
        x_T = x.T
        self.assertTrue(x_T.shape, [7, 9, 6, 3])
        np.testing.assert_array_equal(x_T.numpy(), x_np.T)

        x_np = np.random.randn(3, 6, 9, 7)
        x = paddle.to_tensor(x_np)
        x_mT = x.mT
        self.assertTrue(x_mT.shape, [3, 6, 7, 9])
        np.testing.assert_array_equal(
            x_mT.numpy(), x_np.transpose([0, 1, 3, 2])
        )

        x_np = np.random.randn(3)
        x = paddle.to_tensor(x_np)
        self.assertRaises(ValueError, getattr, x, "mT")

        self.assertTrue(inspect.ismethod(a.dot))
        self.assertTrue(inspect.ismethod(a.logsumexp))
        self.assertTrue(inspect.ismethod(a.multiplex))
        self.assertTrue(inspect.ismethod(a.prod))
        self.assertTrue(inspect.ismethod(a.scale))
        self.assertTrue(inspect.ismethod(a.stanh))
        self.assertTrue(inspect.ismethod(a.add_n))
        self.assertTrue(inspect.ismethod(a.max))
        self.assertTrue(inspect.ismethod(a.maximum))
        self.assertTrue(inspect.ismethod(a.min))
        self.assertTrue(inspect.ismethod(a.minimum))
        self.assertTrue(inspect.ismethod(a.floor_divide))
        self.assertTrue(inspect.ismethod(a.remainder))
        self.assertTrue(inspect.ismethod(a.floor_mod))
        self.assertTrue(inspect.ismethod(a.multiply))
        self.assertTrue(inspect.ismethod(a.inverse))
        self.assertTrue(inspect.ismethod(a.log1p))
        self.assertTrue(inspect.ismethod(a.erf))
        self.assertTrue(inspect.ismethod(a.addmm))
        self.assertTrue(inspect.ismethod(a.clip))
        self.assertTrue(inspect.ismethod(a.trace))
        self.assertTrue(inspect.ismethod(a.kron))
        self.assertTrue(inspect.ismethod(a.isinf))
        self.assertTrue(inspect.ismethod(a.isnan))
        self.assertTrue(inspect.ismethod(a.concat))
        self.assertTrue(inspect.ismethod(a.broadcast_to))
        self.assertTrue(inspect.ismethod(a.scatter_nd_add))
        self.assertTrue(inspect.ismethod(a.scatter_nd))
        self.assertTrue(inspect.ismethod(a.shard_index))
        self.assertTrue(inspect.ismethod(a.chunk))
        self.assertTrue(inspect.ismethod(a.stack))
        self.assertTrue(inspect.ismethod(a.strided_slice))
        self.assertTrue(inspect.ismethod(a.unsqueeze))
        self.assertTrue(inspect.ismethod(a.unstack))
        self.assertTrue(inspect.ismethod(a.argmax))
        self.assertTrue(inspect.ismethod(a.argmin))
        self.assertTrue(inspect.ismethod(a.argsort))
        self.assertTrue(inspect.ismethod(a.masked_select))
        self.assertTrue(inspect.ismethod(a.topk))
        self.assertTrue(inspect.ismethod(a.index_select))
        self.assertTrue(inspect.ismethod(a.nonzero))
        self.assertTrue(inspect.ismethod(a.sort))
        self.assertTrue(inspect.ismethod(a.index_sample))
        self.assertTrue(inspect.ismethod(a.mean))
        self.assertTrue(inspect.ismethod(a.std))
        self.assertTrue(inspect.ismethod(a.numel))
        self.assertTrue(inspect.ismethod(x.asin_))
        self.assertTrue(inspect.ismethod(x.atan2))
        self.assertTrue(inspect.ismethod(x.atanh_))
        self.assertTrue(inspect.ismethod(x.coalesce))
        self.assertTrue(inspect.ismethod(x.diagflat))
        self.assertTrue(inspect.ismethod(x.multinomial))
        self.assertTrue(inspect.ismethod(x.pinv))
        self.assertTrue(inspect.ismethod(x.renorm))
        self.assertTrue(inspect.ismethod(x.renorm_))
        self.assertTrue(inspect.ismethod(x.tan))
        self.assertTrue(inspect.ismethod(x.tan_))
        self.assertTrue(inspect.ismethod(x.tril))
        self.assertTrue(inspect.ismethod(x.tril_))
        self.assertTrue(inspect.ismethod(x.triu))
        self.assertTrue(inspect.ismethod(x.triu_))
        self.assertTrue(inspect.ismethod(x.stft))
        self.assertTrue(inspect.ismethod(x.istft))
        self.assertTrue(inspect.ismethod(x.abs_))
        self.assertTrue(inspect.ismethod(x.acos_))
        self.assertTrue(inspect.ismethod(x.atan_))
        self.assertTrue(inspect.ismethod(x.cos_))
        self.assertTrue(inspect.ismethod(x.cosh_))
        self.assertTrue(inspect.ismethod(x.sin_))
        self.assertTrue(inspect.ismethod(x.sinh_))
        self.assertTrue(inspect.ismethod(x.acosh_))
        self.assertTrue(inspect.ismethod(x.asinh_))
        self.assertTrue(inspect.ismethod(x.diag))

    def test_complex_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with base.dygraph.guard():
            a = paddle.to_tensor(a_np)
            res = 1j * a
            np.testing.assert_array_equal(res.numpy(), 1j * a_np)

    def test_matmul(self):
        x_np = np.random.uniform(-1, 1, [2, 3]).astype(self.dtype)
        y_np = np.random.uniform(-1, 1, [3, 2]).astype(self.dtype)
        except_out = x_np @ y_np

        with base.dygraph.guard():
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = x @ y
            np.testing.assert_allclose(out.numpy(), except_out, atol=1e-03)

    def test_coalesce(self):
        indices = [[0, 0, 1], [1, 1, 2]]
        values = [1.0, 2.0, 3.0]
        sp_x = paddle.sparse.sparse_coo_tensor(indices, values)
        sp_x = sp_x.coalesce()
        self.assertTrue(isinstance(sp_x, paddle.Tensor))


if __name__ == '__main__':
    unittest.main()
