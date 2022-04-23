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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.dygraph as dg
from op_test import OpTest
from paddle.fluid.framework import _test_eager_guard


class TestTensorBackward(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def func_tensor_backward(self):
        for dtype in self._dtypes:
            x = np.random.random([2, 100]).astype(dtype)
            y = np.random.random([100, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor = paddle.matmul(x_tensor, y_tensor)

                    grad_tensor = paddle.to_tensor(grad)
                    z_tensor.backward(grad_tensor)

                    x_grad = np.matmul(grad, y.T)

                    self.assertTrue(np.allclose(x_grad, x_tensor.grad.numpy()))

    def test_tensor_backward(self):
        with _test_eager_guard():
            self.func_tensor_backward()
        self.func_tensor_backward()


class TestBackwardAPI(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def func_backward_api(self):
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)
                    z_tensor2 = paddle.matmul(x_tensor, y_tensor)

                    grad_tensor = paddle.to_tensor(grad)
                    paddle.autograd.backward([z_tensor1, z_tensor2],
                                             [grad_tensor, grad_tensor], True)

                    x_grad = np.matmul(grad, y.T)

                    self.assertTrue(
                        np.allclose(x_grad * 2, x_tensor.grad.numpy()))

    def test_backward_api(self):
        with _test_eager_guard():
            self.func_backward_api()
        self.func_backward_api()

    def func_backward_single_tensor(self):
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)

                    grad_tensor = paddle.to_tensor(grad)
                    paddle.autograd.backward(z_tensor1, grad_tensor, True)

                    x_grad = np.matmul(grad, y.T)

                    self.assertTrue(np.allclose(x_grad, x_tensor.grad.numpy()))

    def test_backward_single_tensor(self):
        with _test_eager_guard():
            self.func_backward_single_tensor()
        self.func_backward_single_tensor()

    def func_backward_none_grad_tensor(self):
        for dtype in self._dtypes:
            x = np.random.random([2, 2]).astype(dtype)
            y = np.random.random([2, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.ones(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor1 = paddle.matmul(x_tensor, y_tensor)

                    paddle.autograd.backward(z_tensor1, None)

                    x_grad = np.matmul(grad, y.T)

                    self.assertTrue(np.allclose(x_grad, x_tensor.grad.numpy()))

    def test_backward_none_grad_tensor(self):
        with _test_eager_guard():
            self.func_backward_none_grad_tensor()
        self.func_backward_none_grad_tensor()

    def func_backward_accumulator_with_init_grad(self):
        for dtype in self._dtypes:
            x = np.random.random([10, ]).astype(dtype)
            y_grad = np.random.random([10, ]).astype(dtype)
            z_grad = np.random.random([10, ]).astype(dtype)
            self._places = [paddle.CPUPlace()]
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = x_tensor**2
                    z_tensor = y_tensor**3

                    y_grad_tensor = paddle.to_tensor(y_grad)
                    z_grad_tensor = paddle.to_tensor(z_grad)
                    paddle.autograd.backward([y_tensor, z_tensor],
                                             [y_grad_tensor, z_grad_tensor])

                    y = x**2
                    z = x**3
                    x_grad = 2 * x * (y_grad + 3 * y * y * z_grad)

                    self.assertTrue(np.allclose(x_grad, x_tensor.grad.numpy()))

    def test_backward_accumulator_with_init_grad(self):
        with _test_eager_guard():
            self.func_backward_accumulator_with_init_grad()
        self.func_backward_accumulator_with_init_grad()


if __name__ == '__main__':
    unittest.main()
