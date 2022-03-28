# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function, division

import unittest
import numpy as np

import paddle
import paddle.fluid.dygraph as dg
from op_test import OpTest
from paddle.fluid.framework import _test_eager_guard


class TestComplexAbsOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = paddle.abs
        self.op_type = "abs"
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype) + 1J * np.random.random(self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype)
        self.grad_x = self.grad_out * (self.x / np.abs(self.x))

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True)


class TestComplexAbsOpZeroValues(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "abs"
        self.python_api = paddle.abs
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        self.x = np.zeros(self.shape).astype(self.dtype) + 1J * np.zeros(
            self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype)
        self.grad_x = np.zeros(self.shape, self.dtype)

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True)


class TestAbs(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_all_positive(self):
        for dtype in self._dtypes:
            x = 1 + 10 * np.random.random([13, 3, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    y = paddle.abs(paddle.to_tensor(x))
                    self.assertTrue(np.allclose(np.abs(x), y.numpy()))

    def test_eager(self):
        with _test_eager_guard():
            self.test_all_positive()


class TestRealAbsOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = paddle.abs
        self.op_type = "abs"
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        self.x = 1 + np.random.random(self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype)
        self.grad_x = self.grad_out * (self.x / np.abs(self.x))

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True)


if __name__ == '__main__':
    unittest.main()
