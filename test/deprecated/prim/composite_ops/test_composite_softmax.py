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
from paddle.base import core, framework
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


def expect_forward(inputs):
    return fn(inputs)


class TestCompositeSoftmax(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = fn(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that softmax in original block
            self.assertTrue('softmax' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that softmax is splitted into small ops
            self.assertTrue('softmax' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        if not attrs.shape and attrs.axis not in [-1, 0]:
            # op softmax does not support both case
            return
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_forward(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_forward()


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False, full_graph=True)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.sf = F.softmax

    def forward(self, x, current_axis):
        out = self.sf(x, axis=current_axis)
        return out


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def train(self, use_prim):
        self.x = paddle.randn(attrs.shape, dtype="float32")
        self.x.stop_gradient = False
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )

        net = paddle.amp.decorate(models=net, level='O2')

        net = apply_to_static(net, False)
        with paddle.amp.auto_cast(level='O2'):
            out = net(self.x, attrs.axis)
            loss = paddle.mean(out)
            grad = paddle.grad(loss, self.x)
            return loss, grad

    def compare_forward(self):
        if not attrs.shape and attrs.axis not in [-1, 0]:
            # op softmax does not support both case
            return
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(
                expected[0],
                actual[0],
                rtol=1e-3,
                atol=1e-3,
            )
            np.testing.assert_allclose(
                expected[1],
                actual[1],
                rtol=1e-3,
                atol=1e-3,
            )

    def test_forward(self):
        for i in self.axes:
            for t in self.shapes:
                attrs.set_axis(i)
                attrs.set_shape(t)
                self.compare_forward()


if __name__ == '__main__':
    unittest.main()
