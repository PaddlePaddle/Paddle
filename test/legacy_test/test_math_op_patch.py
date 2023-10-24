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

import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle
from paddle import base


class TestMathOpPatches(unittest.TestCase):
    @classmethod
    def setUp(self):
        np.random.seed(1024)
        paddle.enable_static()

    @prog_scope()
    def test_add_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = a + 10
        ab = paddle.concat([a, b], axis=1)
        c = ab + 10
        d = ab + a
        # e = a + ab
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        b_np, c_np, d_np = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b, c, d]
        )
        np.testing.assert_allclose(a_np + 10, b_np, rtol=1e-05)
        ab_np = np.concatenate([a_np, b_np], axis=1)
        np.testing.assert_allclose(ab_np + 10, c_np, rtol=1e-05)
        d_expected = ab_np + np.concatenate([a_np, a_np], axis=1)
        np.testing.assert_allclose(d_expected, d_np, rtol=1e-05)

    @prog_scope()
    def test_radd_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = 10 + a
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(a_np + 10, b_np, rtol=1e-05)

    @prog_scope()
    def test_sub_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = a - 10
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(a_np - 10, b_np, rtol=1e-05)

    @prog_scope()
    def test_rsub_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = 10 - a
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(10 - a_np, b_np, rtol=1e-05)

    @prog_scope()
    def test_mul_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = a * 10
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(a_np * 10, b_np, rtol=1e-05)

    @prog_scope()
    def test_rmul_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = 10 * a
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(10 * a_np, b_np, rtol=1e-05)

    @prog_scope()
    def test_div_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = a / 10
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(a_np / 10, b_np, rtol=1e-05)

    @prog_scope()
    def test_rdiv_scalar(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = 10 / a
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32') + 1e-2

        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(10 / a_np, b_np, rtol=1e-05)

    @prog_scope()
    def test_div_two_tensor(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = paddle.static.data(name="b", shape=[-1, 1])
        c = a / b
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        b_np = np.random.random(size=[10, 1]).astype('float32') + 1e-2
        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, 'b': b_np},
            fetch_list=[c],
        )
        np.testing.assert_allclose(a_np / b_np, c_np, rtol=1e-05)

    @prog_scope()
    def test_mul_two_tensor(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = paddle.static.data(name="b", shape=[-1, 1])
        c = a * b
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        b_np = np.random.random(size=[10, 1]).astype('float32')
        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, 'b': b_np},
            fetch_list=[c],
        )
        np.testing.assert_allclose(a_np * b_np, c_np, rtol=1e-05)

    @prog_scope()
    def test_add_two_tensor(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = paddle.static.data(name="b", shape=[-1, 1])
        c = a + b
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        b_np = np.random.random(size=[10, 1]).astype('float32')
        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, 'b': b_np},
            fetch_list=[c],
        )
        np.testing.assert_allclose(a_np + b_np, c_np, rtol=1e-05)

    @prog_scope()
    def test_sub_two_tensor(self):
        a = paddle.static.data(name="a", shape=[-1, 1])
        b = paddle.static.data(name="b", shape=[-1, 1])
        c = a - b
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.random(size=[10, 1]).astype('float32')
        b_np = np.random.random(size=[10, 1]).astype('float32')
        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, 'b': b_np},
            fetch_list=[c],
        )
        np.testing.assert_allclose(a_np - b_np, c_np, rtol=1e-05)

    @prog_scope()
    def test_integer_div(self):
        a = paddle.static.data(name="a", shape=[-1, 1], dtype='int64')
        b = a / 7
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.array([3, 4, 10, 14, 9, 18]).astype('int64')
        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )

        b_np_actual = (a_np / 7).astype('float32')
        np.testing.assert_allclose(b_np, b_np_actual, rtol=1e-05)

    @prog_scope()
    def test_equal(self):
        a = paddle.static.data(name="a", shape=[-1, 1], dtype='float32')
        b = paddle.static.data(name="b", shape=[-1, 1], dtype='float32')
        c = a == b

        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        b_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')

        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, "b": b_np},
            fetch_list=[c],
        )

        np.testing.assert_array_equal(c_np, a_np == b_np)
        self.assertEqual(c.dtype, base.core.VarDesc.VarType.BOOL)

    @prog_scope()
    def test_equal_and_cond(self):
        a = paddle.static.data(name="a", shape=[-1, 1], dtype='float32')
        a.desc.set_need_check_feed(False)
        b = paddle.static.data(name="b", shape=[-1, 1], dtype='float32')
        b.desc.set_need_check_feed(False)
        one = paddle.ones(shape=[1], dtype='int32')
        zero = paddle.zeros(shape=[1], dtype='int32')
        cond = one == zero
        c = paddle.static.nn.cond(cond, lambda: a + b, lambda: a - b)

        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        b_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')
        (c_np,) = exe.run(
            base.default_main_program(),
            feed={"a": a_np, "b": b_np},
            fetch_list=[c],
        )

        np.testing.assert_array_equal(c_np, a_np - b_np)

    @prog_scope()
    def test_neg(self):
        a = paddle.static.data(name="a", shape=[-1, 10, 1], dtype='float32')
        a.desc.set_need_check_feed(False)
        b = -a
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.uniform(-1, 1, size=[10, 1]).astype('float32')

        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(-a_np, b_np, rtol=1e-05)

    @prog_scope()
    def test_astype(self):
        a = paddle.static.data(name="a", shape=[-1, 10, 1])
        a.desc.set_need_check_feed(False)
        b = a.astype('float32')
        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.random.uniform(-1, 1, size=[10, 1]).astype('float64')

        (b_np,) = exe.run(
            base.default_main_program(), feed={"a": a_np}, fetch_list=[b]
        )
        np.testing.assert_allclose(a_np.astype('float32'), b_np, rtol=1e-05)

    def test_bitwise_and(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np & y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x & y

        exe = base.Executor()
        out = exe.run(
            base.default_main_program(),
            feed={"x": x_np, "y": y_np},
            fetch_list=[z],
        )
        np.testing.assert_array_equal(out[0], out_np)

    @prog_scope()
    def test_bitwise_or(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np | y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x | y

        exe = base.Executor()
        out = exe.run(
            base.default_main_program(),
            feed={"x": x_np, "y": y_np},
            fetch_list=[z],
        )
        np.testing.assert_array_equal(out[0], out_np)

    @prog_scope()
    def test_bitwise_xor(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np ^ y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x ^ y

        exe = base.Executor()
        out = exe.run(
            base.default_main_program(),
            feed={"x": x_np, "y": y_np},
            fetch_list=[z],
        )
        np.testing.assert_array_equal(out[0], out_np)

    @prog_scope()
    def test_bitwise_not(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = ~x_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        z = ~x

        exe = base.Executor()
        out = exe.run(
            base.default_main_program(), feed={"x": x_np}, fetch_list=[z]
        )
        np.testing.assert_array_equal(out[0], out_np)

    @prog_scope()
    def test_T(self):
        x_np = np.random.randint(-100, 100, [2, 8, 5, 3]).astype("int32")
        out_np = x_np.T

        x = paddle.static.data(name="x", shape=[2, 8, 5, 3], dtype="int32")
        z = x.T

        exe = base.Executor()
        out = exe.run(
            base.default_main_program(), feed={"x": x_np}, fetch_list=[z]
        )
        np.testing.assert_array_equal(out[0], out_np)

    @prog_scope()
    def test_ndim(self):
        a = paddle.static.data(name="a", shape=[10, 1])
        self.assertEqual(a.dim(), 2)
        self.assertEqual(a.ndimension(), 2)
        self.assertEqual(a.ndim, 2)

    @prog_scope()
    def test_matmul(self):
        a = paddle.static.data(name='a', shape=[2, 3], dtype='float32')
        b = paddle.static.data(name='b', shape=[3, 5], dtype='float32')
        c = a @ b  # __matmul__
        a_np = np.random.uniform(-1, 1, size=[2, 3]).astype('float32')
        b_np = np.random.uniform(-1, 1, size=[3, 5]).astype('float32')
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        (c_np,) = exe.run(
            paddle.static.default_main_program(),
            feed={"a": a_np, "b": b_np},
            fetch_list=[c],
        )
        np.testing.assert_allclose(a_np @ b_np, c_np, rtol=1e-05)


class TestDygraphMathOpPatches(unittest.TestCase):
    def init_data(self):
        self.np_a = np.random.random((2, 3, 4)).astype(np.float32)
        self.np_b = np.random.random((2, 3, 4)).astype(np.float32)
        self.np_a[np.abs(self.np_a) < 0.0005] = 0.002
        self.np_b[np.abs(self.np_b) < 0.0005] = 0.002

        self.tensor_a = paddle.to_tensor(self.np_a, dtype="float32")
        self.tensor_b = paddle.to_tensor(self.np_b, dtype="float32")

    def test_dygraph_greater_than(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor > nparray
        expect_out = self.np_a > self.np_b
        actual_out = self.tensor_a > self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_greater_equal(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor >= nparray
        expect_out = self.np_a >= self.np_b
        actual_out = self.tensor_a >= self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_reminder(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor % nparray
        expect_out = self.np_a % self.np_b
        actual_out = self.tensor_a % self.np_b
        np.testing.assert_allclose(actual_out, expect_out, rtol=1e-7, atol=1e-7)
        paddle.enable_static()

    def test_dygraph_less_than(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor < nparray
        expect_out = self.np_a < self.np_b
        actual_out = self.tensor_a < self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_less_equal(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor <= nparray
        expect_out = self.np_a <= self.np_b
        actual_out = self.tensor_a <= self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_floor_divide(self):
        paddle.disable_static()
        np_a = np.random.random((2, 3, 4)).astype(np.int32)
        np_b = np.random.random((2, 3, 4)).astype(np.int32)
        np_b[np.abs(np_b) < 1] = 2
        # normal case: tenor // nparray
        tensor_a = paddle.to_tensor(np_a, dtype="int32")
        tensor_b = paddle.to_tensor(np_b, dtype="int32")
        expect_out = np_a // np_b
        actual_out = tensor_a // np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_elementwise_pow(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor ** nparray
        expect_out = self.np_a**self.np_b
        actual_out = self.tensor_a**self.np_b
        np.testing.assert_allclose(actual_out, expect_out, rtol=1e-7, atol=1e-7)

        # normal case: nparray ** tensor
        expect_out = self.np_a**self.np_b
        actual_out = self.np_a**self.tensor_b
        np.testing.assert_allclose(actual_out, expect_out, rtol=1e-7, atol=1e-7)

        paddle.enable_static()

    def test_dygraph_not_equal(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor != nparray
        expect_out = self.np_a != self.np_b
        actual_out = self.tensor_a != self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()

    def test_dygraph_equal(self):
        paddle.disable_static()
        self.init_data()
        # normal case: tenor == nparray
        expect_out = self.np_a == self.np_b
        actual_out = self.tensor_a == self.np_b
        np.testing.assert_equal(actual_out, expect_out)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
