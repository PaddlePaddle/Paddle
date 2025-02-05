# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from prim.composite_ops.utils import TOLERANCE

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.autograd import primapi


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = None
        self.axis = -1
        self.shape = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def set_axis(self, axis) -> None:
        self.axis = axis

    def set_shape(self, shape) -> None:
        self.shape = shape

    def get_rtol(self, flag):
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x):
    return F.softmax(x, axis=attrs.axis, dtype=attrs.dtype)


def expect_grad(inputs):
    paddle.disable_static()
    inputs.stop_gradient = False
    res = fn(inputs)

    gradients = paddle.grad(res, inputs)
    return gradients


class TestCompositeSoftmax(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.shapes = [[2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite_grad(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            y = fn(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that softmax in original block
            self.assertTrue('softmax' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that softmax is splitted into small ops
            self.assertTrue('softmax' not in fwd_ops_new)

            z = paddle.static.gradients([y], x)
            fwd_ops_grad = [op.type for op in blocks[0].ops]
            # Ensure that softmax_grad not in grad block

            self.assertTrue('softmax_grad' not in fwd_ops_grad)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_backward(self):
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("backward"),
            atol=attrs.get_atol("backward"),
        )

    def test_backward(self):
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()


class TestCompositeSoftmaxPrimBackward(unittest.TestCase):
    "test composite softmax and prim backward"

    def setUp(self):
        core._set_prim_backward_enabled(True)
        self.dtypes = ["float32", "float64"]
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite_grad(self, inputs):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            y = fn(x)
            blocks = main_program.blocks
            primapi.to_prim(blocks)
            z = paddle.static.gradients([y], x)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def compare_backward(self):
        if not attrs.shape and attrs.axis not in [-1, 0]:
            # op softmax does not support both case
            return
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("prim_backward"),
            atol=attrs.get_rtol("prim_backward"),
        )

    def test_prim_backward(self):
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()


if __name__ == '__main__':
    unittest.main()
