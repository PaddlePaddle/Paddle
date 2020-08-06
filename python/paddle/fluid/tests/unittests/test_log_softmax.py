#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
import paddle.nn.functional as F

np.random.seed(10)


def ref_log_softmax(x):
    shiftx = (x - np.max(x))
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full(x.shape, fill_value=1 / x.size, dtype='float64')
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis)
    return dx


class TestLogSoftmaxOp(OpTest):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = 'log_softmax'
        self.dtype = 'float64'
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        x = np.random.uniform(0.1, 1., self.shape).astype('float64')
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad])


class TestLogSoftmaxShape(TestLogSoftmaxOp):
    def get_x_shape(self):
        return [12, 10]


class TestLogSoftmaxAxis(TestLogSoftmaxOp):
    def get_x_axis(self):
        return 1


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1., 1., self.x_shape).astype(np.float32)
        self.place = paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def check_api(self, axis=-1):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)

        main_program = paddle.Program()
        logsoftmax = paddle.nn.LogSoftmax(axis)
        with paddle.program_guard(main_program):
            x = paddle.data(name='x', shape=self.x_shape)
            y = logsoftmax(x)
        exe = paddle.Executor(self.place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        with paddle.imperative.guard(self.place):
            x = paddle.imperative.to_variable(self.x)
            y = logsoftmax(x)
        self.assertTrue(np.allclose(y.numpy(), ref_out))

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def check_api(self, axis=-1, dtype=None):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)
        main_program = paddle.Program()
        with paddle.program_guard(main_program):
            x = paddle.data(name='x', shape=self.x_shape)
            y = F.log_softmax(x, axis, dtype)
        exe = paddle.Executor(self.place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        with paddle.imperative.guard(self.place):
            x = paddle.imperative.to_variable(self.x)
            y = F.log_softmax(x, axis, dtype)
        self.assertTrue(np.allclose(y.numpy(), ref_out))

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)
        self.check_api(-1, 'float64')

    def test_errors(self):
        with paddle.program_guard(paddle.Program(), paddle.Program()):
            x = paddle.data(name='X1', shape=[100], dtype='int32')
            self.assertRaises(TypeError, F.log_softmax, x)

            x = paddle.data(name='X2', shape=[100], dtype='float32')
            self.assertRaises(TypeError, F.log_softmax, x, dtype='int32')


if __name__ == "__main__":
    unittest.main()
