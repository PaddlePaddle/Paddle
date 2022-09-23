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

from __future__ import print_function

import unittest
import paddle
import paddle.fluid as fluid
import numpy as np
import inspect
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph


class TestMathOpPatchesVarBase(unittest.TestCase):

    def setUp(self):
        self.shape = [10, 1024]
        self.dtype = np.float32

    def func_test_add(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_add(self):
        with _test_eager_guard():
            self.func_test_add()
        self.func_test_add()

    def func_test_sub(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a - b
            np.testing.assert_array_equal(res.numpy(), a_np - b_np)

    def test_sub(self):
        with _test_eager_guard():
            self.func_test_sub()
        self.func_test_sub()

    def func_test_mul(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a * b
            np.testing.assert_array_equal(res.numpy(), a_np * b_np)

    def test_mul(self):
        with _test_eager_guard():
            self.func_test_mul()
        self.func_test_mul()

    def func_test_div(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a / b
            #NOTE: Not sure why array_equal fails on windows, allclose is acceptable
            np.testing.assert_allclose(res.numpy(), a_np / b_np, rtol=1e-05)

    def test_div(self):
        with _test_eager_guard():
            self.func_test_div()
        self.func_test_div()

    def func_test_add_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b)

    def test_add_scalar(self):
        with _test_eager_guard():
            self.func_test_add_scalar()
        self.func_test_add_scalar()

    def func_test_add_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = b + a
            np.testing.assert_array_equal(res.numpy(), b + a_np)

    def test_add_scalar_reverse(self):
        with _test_eager_guard():
            self.func_test_add_scalar_reverse()
        self.func_test_add_scalar_reverse()

    def func_test_sub_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a - b
            np.testing.assert_array_equal(res.numpy(), a_np - b)

    def test_sub_scalar(self):
        with _test_eager_guard():
            self.func_test_sub_scalar()
        self.func_test_sub_scalar()

    def func_test_sub_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = b - a
            np.testing.assert_array_equal(res.numpy(), b - a_np)

    def test_sub_scalar_reverse(self):
        with _test_eager_guard():
            self.func_test_sub_scalar_reverse()
        self.func_test_sub_scalar_reverse()

    def func_test_mul_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a * b
            np.testing.assert_array_equal(res.numpy(), a_np * b)

    def test_mul_scalar(self):
        with _test_eager_guard():
            self.func_test_mul_scalar()
        self.func_test_mul_scalar()

    # div_scalar, not equal
    def func_test_div_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a / b
            np.testing.assert_allclose(res.numpy(), a_np / b, rtol=1e-05)

    def test_div_scalar(self):
        with _test_eager_guard():
            self.func_test_div_scalar()
        self.func_test_div_scalar()

    # pow of float type, not equal
    def func_test_pow(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a**b
            np.testing.assert_allclose(res.numpy(), a_np**b_np, rtol=1e-05)

    def test_pow(self):
        with _test_eager_guard():
            self.func_test_pow()
        self.func_test_pow()

    def func_test_floor_div(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a // b
            np.testing.assert_array_equal(res.numpy(), a_np // b_np)

    def test_floor_div(self):
        with _test_eager_guard():
            self.func_test_floor_div()
        self.func_test_floor_div()

    def func_test_mod(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a % b
            np.testing.assert_array_equal(res.numpy(), a_np % b_np)

    def test_mod(self):
        with _test_eager_guard():
            self.func_test_mod()
        self.func_test_mod()

    # for bitwise and/or/xor/not
    def func_test_bitwise(self):
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

    def test_bitwise(self):
        with _test_eager_guard():
            self.func_test_bitwise()
        self.func_test_bitwise()

    # for logical compare
    def func_test_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            c = fluid.dygraph.to_variable(c_np)
            res1 = (a == b)
            res2 = (a == c)
            np.testing.assert_array_equal(res1.numpy(), a_np == b_np)
            np.testing.assert_array_equal(res2.numpy(), a_np == c_np)

    def test_equal(self):
        with _test_eager_guard():
            self.func_test_equal()
        self.func_test_equal()

    def func_test_not_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            c = fluid.dygraph.to_variable(c_np)
            res1 = (a != b)
            res2 = (a != c)
            np.testing.assert_array_equal(res1.numpy(), a_np != b_np)
            np.testing.assert_array_equal(res2.numpy(), a_np != c_np)

    def test_not_equal(self):
        with _test_eager_guard():
            self.func_test_not_equal()
        self.func_test_not_equal()

    def func_test_less_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a < b)
            np.testing.assert_array_equal(res.numpy(), a_np < b_np)

    def test_less_than(self):
        with _test_eager_guard():
            self.func_test_less_than()
        self.func_test_less_than()

    def func_test_less_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a <= b)
            np.testing.assert_array_equal(res.numpy(), a_np <= b_np)

    def test_less_equal(self):
        with _test_eager_guard():
            self.func_test_less_equal()
        self.func_test_less_equal()

    def func_test_greater_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a > b)
            np.testing.assert_array_equal(res.numpy(), a_np > b_np)

    def test_greater_than(self):
        with _test_eager_guard():
            self.func_test_greater_than()
        self.func_test_greater_than()

    def func_test_greater_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a >= b)
            np.testing.assert_array_equal(res.numpy(), a_np >= b_np)

    def test_greater_equal(self):
        with _test_eager_guard():
            self.func_test_greater_equal()
        self.func_test_greater_equal()

    def func_test_neg(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            res = -a
            np.testing.assert_array_equal(res.numpy(), -a_np)

    def test_neg(self):
        with _test_eager_guard():
            self.func_test_neg()
        self.func_test_neg()

    def func_test_float_int_long(self):
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(np.array([100.1]))
            self.assertTrue(float(a) == 100.1)
            self.assertTrue(int(a) == 100)
            self.assertTrue(int(a) == 100)

    def test_float_int_long(self):
        with _test_eager_guard():
            self.func_test_float_int_long()
        self.func_test_float_int_long()

    def func_test_len(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            self.assertTrue(len(a) == 10)

    def test_len(self):
        with _test_eager_guard():
            self.func_test_len()
        self.func_test_len()

    def func_test_index(self):
        with fluid.dygraph.guard():
            var1 = fluid.dygraph.to_variable(np.array([2]))
            i_tmp = 0
            for i in range(var1):
                self.assertTrue(i == i_tmp)
                i_tmp = i_tmp + 1
            list1 = [1, 2, 3, 4, 5]
            self.assertTrue(list1[var1] == 3)
            str1 = "just test"
            self.assertTrue(str1[var1] == 's')

    def test_index(self):
        with _test_eager_guard():
            self.func_test_index()
        self.func_test_index()

    def func_test_np_left_mul(self):
        with fluid.dygraph.guard():
            t = np.sqrt(2.0 * np.pi)
            x = fluid.layers.ones((2, 2), dtype="float32")
            y = t * x

            np.testing.assert_allclose(y.numpy(),
                                       t * np.ones((2, 2), dtype='float32'),
                                       rtol=1e-05,
                                       atol=0.0)

    def test_np_left_mul(self):
        with _test_eager_guard():
            self.func_test_np_left_mul()
        self.func_test_np_left_mul()

    def func_test_add_different_dtype(self):
        a_np = np.random.random(self.shape).astype(np.float32)
        b_np = np.random.random(self.shape).astype(np.float16)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a + b
            np.testing.assert_array_equal(res.numpy(), a_np + b_np)

    def test_add_different_dtype(self):
        with _test_eager_guard():
            self.func_test_add_different_dtype()
        self.func_test_add_different_dtype()

    def func_test_floordiv_different_dtype(self):
        a_np = np.full(self.shape, 10, np.int64)
        b_np = np.full(self.shape, 2, np.int32)
        with fluid.dygraph.guard():
            a = paddle.to_tensor(a_np)
            b = paddle.to_tensor(b_np)
            res = a // b
            np.testing.assert_array_equal(res.numpy(), a_np // b_np)

    def test_floordiv_different_dtype(self):
        with _test_eager_guard():
            self.func_test_floordiv_different_dtype()
        self.func_test_floordiv_different_dtype()

    def func_test_astype(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            res1 = a.astype(np.float16)
            res2 = a.astype('float16')
            res3 = a.astype(fluid.core.VarDesc.VarType.FP16)

            self.assertEqual(res1.dtype, res2.dtype)
            self.assertEqual(res1.dtype, res3.dtype)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())
            np.testing.assert_array_equal(res1.numpy(), res3.numpy())

    def test_astype(self):
        with _test_eager_guard():
            self.func_test_astype()
        self.func_test_astype()

    def func_test_conpare_op_broadcast(self):
        a_np = np.random.uniform(-1, 1, [10, 1, 10]).astype(self.dtype)
        b_np = np.random.uniform(-1, 1, [1, 1, 10]).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)

            self.assertEqual((a != b).dtype, fluid.core.VarDesc.VarType.BOOL)
            np.testing.assert_array_equal((a != b).numpy(), a_np != b_np)

    def test_conpare_op_broadcast(self):
        with _test_eager_guard():
            self.func_test_conpare_op_broadcast()
        self.func_test_conpare_op_broadcast()

    def func_test_tensor_patch_method(self):
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
        np.testing.assert_array_equal(m.rsqrt().numpy(),
                                      paddle.rsqrt(m).numpy())
        np.testing.assert_array_equal(x.ceil().numpy(), paddle.ceil(x).numpy())
        np.testing.assert_array_equal(x.floor().numpy(),
                                      paddle.floor(x).numpy())
        np.testing.assert_array_equal(x.cos().numpy(), paddle.cos(x).numpy())
        np.testing.assert_array_equal(x.acos().numpy(), paddle.acos(x).numpy())
        np.testing.assert_array_equal(x.asin().numpy(), paddle.asin(x).numpy())
        np.testing.assert_array_equal(x.sin().numpy(), paddle.sin(x).numpy())
        np.testing.assert_array_equal(x.sinh().numpy(), paddle.sinh(x).numpy())
        np.testing.assert_array_equal(x.cosh().numpy(), paddle.cosh(x).numpy())
        np.testing.assert_array_equal(x.round().numpy(),
                                      paddle.round(x).numpy())
        np.testing.assert_array_equal(x.reciprocal().numpy(),
                                      paddle.reciprocal(x).numpy())
        np.testing.assert_array_equal(x.square().numpy(),
                                      paddle.square(x).numpy())
        np.testing.assert_array_equal(x.rank().numpy(), paddle.rank(x).numpy())
        np.testing.assert_array_equal(x[0].t().numpy(), paddle.t(x[0]).numpy())
        np.testing.assert_array_equal(x.asinh().numpy(),
                                      paddle.asinh(x).numpy())
        ### acosh(x) = nan, need to change input
        t_np = np.random.uniform(1, 2, [2, 3]).astype(self.dtype)
        t = paddle.to_tensor(t_np)
        np.testing.assert_array_equal(t.acosh().numpy(),
                                      paddle.acosh(t).numpy())
        np.testing.assert_array_equal(x.atanh().numpy(),
                                      paddle.atanh(x).numpy())
        d = paddle.to_tensor([[1.2285208, 1.3491015, 1.4899898],
                              [1.30058, 1.0688717, 1.4928783],
                              [1.0958099, 1.3724753, 1.8926544]])
        d = d.matmul(d.t())
        # ROCM not support cholesky
        if not fluid.core.is_compiled_with_rocm():
            np.testing.assert_array_equal(d.cholesky().numpy(),
                                          paddle.cholesky(d).numpy())

        np.testing.assert_array_equal(x.is_empty().numpy(),
                                      paddle.is_empty(x).numpy())
        np.testing.assert_array_equal(x.isfinite().numpy(),
                                      paddle.isfinite(x).numpy())
        np.testing.assert_array_equal(
            x.cast('int32').numpy(),
            paddle.cast(x, 'int32').numpy())
        np.testing.assert_array_equal(
            x.expand([3, 2, 3]).numpy(),
            paddle.expand(x, [3, 2, 3]).numpy())
        np.testing.assert_array_equal(
            x.tile([2, 2]).numpy(),
            paddle.tile(x, [2, 2]).numpy())
        np.testing.assert_array_equal(x.flatten().numpy(),
                                      paddle.flatten(x).numpy())
        index = paddle.to_tensor([0, 1])
        np.testing.assert_array_equal(
            x.gather(index).numpy(),
            paddle.gather(x, index).numpy())
        index = paddle.to_tensor([[0, 1], [1, 2]])
        np.testing.assert_array_equal(
            x.gather_nd(index).numpy(),
            paddle.gather_nd(x, index).numpy())
        np.testing.assert_array_equal(
            x.reverse([0, 1]).numpy(),
            paddle.reverse(x, [0, 1]).numpy())
        np.testing.assert_array_equal(
            a.reshape([3, 2]).numpy(),
            paddle.reshape(a, [3, 2]).numpy())
        np.testing.assert_array_equal(
            x.slice([0, 1], [0, 0], [1, 2]).numpy(),
            paddle.slice(x, [0, 1], [0, 0], [1, 2]).numpy())
        np.testing.assert_array_equal(
            x.split(2)[0].numpy(),
            paddle.split(x, 2)[0].numpy())
        m = paddle.to_tensor(
            np.random.uniform(-1, 1, [1, 6, 1, 1]).astype(self.dtype))
        np.testing.assert_array_equal(
            m.squeeze([]).numpy(),
            paddle.squeeze(m, []).numpy())
        np.testing.assert_array_equal(
            m.squeeze([1, 2]).numpy(),
            paddle.squeeze(m, [1, 2]).numpy())
        m = paddle.to_tensor([2, 3, 3, 1, 5, 3], 'float32')
        np.testing.assert_array_equal(m.unique()[0].numpy(),
                                      paddle.unique(m)[0].numpy())
        np.testing.assert_array_equal(
            m.unique(return_counts=True)[1],
            paddle.unique(m, return_counts=True)[1])
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
            x.divide(y).numpy(),
            paddle.divide(x, y).numpy())
        np.testing.assert_array_equal(
            x.matmul(y, True, False).numpy(),
            paddle.matmul(x, y, True, False).numpy())
        np.testing.assert_array_equal(
            x.norm(p='fro', axis=[0, 1]).numpy(),
            paddle.norm(x, p='fro', axis=[0, 1]).numpy())
        np.testing.assert_array_equal(
            x.dist(y).numpy(),
            paddle.dist(x, y).numpy())
        np.testing.assert_array_equal(
            x.cross(y).numpy(),
            paddle.cross(x, y).numpy())
        m = x.expand([2, 2, 3])
        n = y.expand([2, 2, 3]).transpose([0, 2, 1])
        np.testing.assert_array_equal(
            m.bmm(n).numpy(),
            paddle.bmm(m, n).numpy())
        np.testing.assert_array_equal(
            x.histogram(5, -1, 1).numpy(),
            paddle.histogram(x, 5, -1, 1).numpy())
        np.testing.assert_array_equal(
            x.equal(y).numpy(),
            paddle.equal(x, y).numpy())
        np.testing.assert_array_equal(
            x.greater_equal(y).numpy(),
            paddle.greater_equal(x, y).numpy())
        np.testing.assert_array_equal(
            x.greater_than(y).numpy(),
            paddle.greater_than(x, y).numpy())
        np.testing.assert_array_equal(
            x.less_equal(y).numpy(),
            paddle.less_equal(x, y).numpy())
        np.testing.assert_array_equal(
            x.less_than(y).numpy(),
            paddle.less_than(x, y).numpy())
        np.testing.assert_array_equal(
            x.not_equal(y).numpy(),
            paddle.not_equal(x, y).numpy())
        np.testing.assert_array_equal(
            x.equal_all(y).numpy(),
            paddle.equal_all(x, y).numpy())
        np.testing.assert_array_equal(
            x.allclose(y).numpy(),
            paddle.allclose(x, y).numpy())
        m = x.expand([2, 2, 3])
        np.testing.assert_array_equal(
            x.expand_as(m).numpy(),
            paddle.expand_as(x, m).numpy())
        index = paddle.to_tensor([2, 1, 0])
        np.testing.assert_array_equal(
            a.scatter(index, b).numpy(),
            paddle.scatter(a, index, b).numpy())

        # 3. Bool tensor operation
        x = paddle.to_tensor([[True, False], [True, False]])
        y = paddle.to_tensor([[False, False], [False, True]])
        np.testing.assert_array_equal(
            x.logical_and(y).numpy(),
            paddle.logical_and(x, y).numpy())
        np.testing.assert_array_equal(
            x.logical_not(y).numpy(),
            paddle.logical_not(x, y).numpy())
        np.testing.assert_array_equal(
            x.logical_or(y).numpy(),
            paddle.logical_or(x, y).numpy())
        np.testing.assert_array_equal(
            x.logical_xor(y).numpy(),
            paddle.logical_xor(x, y).numpy())
        np.testing.assert_array_equal(
            x.logical_and(y).numpy(),
            paddle.logical_and(x, y).numpy())
        a = paddle.to_tensor([[1, 2], [3, 4]])
        b = paddle.to_tensor([[4, 3], [2, 1]])
        np.testing.assert_array_equal(
            x.where(a, b).numpy(),
            paddle.where(x, a, b).numpy())

        x_np = np.random.randn(3, 6, 9, 7)
        x = paddle.to_tensor(x_np)
        x_T = x.T
        self.assertTrue(x_T.shape, [7, 9, 6, 3])
        np.testing.assert_array_equal(x_T.numpy(), x_np.T)

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
        self.assertTrue(inspect.ismethod(a.logsumexp))
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

    def test_tensor_patch_method(self):
        with _test_eager_guard():
            self.func_test_tensor_patch_method()
        self.func_test_tensor_patch_method()

    def func_test_complex_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            res = 1J * a
            np.testing.assert_array_equal(res.numpy(), 1j * a_np)

    def test_complex_scalar(self):
        with _test_eager_guard():
            self.func_test_complex_scalar()
        self.func_test_complex_scalar()


if __name__ == '__main__':
    unittest.main()
