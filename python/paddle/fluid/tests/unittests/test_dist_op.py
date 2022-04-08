# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()


def dist(x, y, p):
    if p == 0.:
        out = np.count_nonzero(x - y)
    elif p == float("inf"):
        out = np.max(np.abs(x - y))
    elif p == float("-inf"):
        out = np.min(np.abs(x - y))
    else:
        out = np.power(np.sum(np.power(np.abs(x - y), p)), 1.0 / p)
    return np.array(out).astype(x.dtype)


class TestDistOp(OpTest):
    def setUp(self):
        self.op_type = 'dist'
        self.python_api = paddle.dist
        self.attrs = {}
        self.init_case()
        self.init_data_type()
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.data_type),
            "Y": np.random.random(self.y_shape).astype(self.data_type)
        }

        self.attrs["p"] = self.p
        self.outputs = {
            "Out": dist(self.inputs["X"], self.inputs["Y"], self.attrs["p"])
        }
        self.gradient = self.calc_gradient()

    def init_case(self):
        self.x_shape = (120)
        self.y_shape = (120)
        self.p = 0.

    def init_data_type(self):
        self.data_type = np.float32 if core.is_compiled_with_rocm(
        ) else np.float64

    def calc_gradient(self):
        x = self.inputs["X"]
        y = self.inputs["Y"]
        p = self.attrs["p"]
        if p == 0:
            grad = np.zeros(x.shape)
        elif p in [float("inf"), float("-inf")]:
            norm = dist(x, y, p)
            x_minux_y_abs = np.abs(x - y)
            grad = np.sign(x - y)
            grad[x_minux_y_abs != norm] = 0
        else:
            norm = dist(x, y, p)
            grad = np.power(norm, 1 - p) * np.power(np.abs(x - y),
                                                    p - 1) * np.sign(x - y)

        def get_reduce_dims(x, y):
            x_reduce_dims = []
            y_reduce_dims = []

            if x.ndim >= y.ndim:
                y_reshape = tuple([1] * (x.ndim - y.ndim) + list(y.shape))
                y = y.reshape(y_reshape)
            else:
                x_reshape = tuple([1] * (y.ndim - x.ndim) + list(x.shape))
                x = x.reshape(x_reshape)
            for i in range(x.ndim):
                if x.shape[i] > y.shape[i]:
                    y_reduce_dims.append(i)
                elif x.shape[i] < y.shape[i]:
                    x_reduce_dims.append(i)
            return x_reduce_dims, y_reduce_dims

        x_reduce_dims, y_reduce_dims = get_reduce_dims(x, y)
        if len(x_reduce_dims) != 0:
            x_grad = np.sum(grad, tuple(x_reduce_dims)).reshape(x.shape)
        else:
            x_grad = grad
        if len(y_reduce_dims) != 0:
            y_grad = -np.sum(grad, tuple(y_reduce_dims)).reshape(y.shape)
        else:
            y_grad = -grad

        return x_grad, y_grad

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(
            ["X", "Y"],
            "Out",
            user_defined_grads=self.gradient,
            check_eager=True)


class TestDistOpCase1(TestDistOp):
    def init_case(self):
        self.x_shape = (3, 5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 1.


class TestDistOpCase2(TestDistOp):
    def init_case(self):
        self.x_shape = (10, 10)
        self.y_shape = (4, 10, 10)
        self.p = 2.


class TestDistOpCase3(TestDistOp):
    def init_case(self):
        self.x_shape = (15, 10)
        self.y_shape = (15, 10)
        self.p = float("inf")


class TestDistOpCase4(TestDistOp):
    def init_case(self):
        self.x_shape = (2, 3, 4, 5, 8)
        self.y_shape = (3, 1, 5, 8)
        self.p = float("-inf")


class TestDistOpCase5(TestDistOp):
    def init_case(self):
        self.x_shape = (4, 1, 4, 8)
        self.y_shape = (2, 2, 1, 4, 4, 8)
        self.p = 1.5


class TestDistAPI(unittest.TestCase):
    def init_data_type(self):
        self.data_type = 'float32' if core.is_compiled_with_rocm(
        ) else 'float64'

    def test_api(self):
        self.init_data_type()
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.data(name='x', shape=[2, 3, 4, 5], dtype=self.data_type)
            y = fluid.data(name='y', shape=[3, 1, 5], dtype=self.data_type)
            p = 2
            x_i = np.random.random((2, 3, 4, 5)).astype(self.data_type)
            y_i = np.random.random((3, 1, 5)).astype(self.data_type)
            result = paddle.dist(x, y, p)
            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(fluid.default_main_program(),
                          feed={'x': x_i,
                                'y': y_i},
                          fetch_list=[result])
            self.assertTrue(np.allclose(dist(x_i, y_i, p), out[0]))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
