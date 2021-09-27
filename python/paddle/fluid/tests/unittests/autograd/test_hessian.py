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

import unittest
import numpy as np
import paddle
import paddle.compat as cpt
from utils import _compute_numerical_hessian


class TestHessian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-2
        self.rtol = 1e-2
        self.atol = 1e-2
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_hessian = _compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(func, [self.x, self.y])
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                assert np.allclose(hessian[i][j].numpy(),
                                   numerical_hessian[i][j], self.rtol,
                                   self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            hessian = paddle.autograd.hessian(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(
            func, [self.x, self.y], allow_unused=True)
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                if i == j == 0:
                    assert np.allclose(hessian[i][j].numpy(),
                                       numerical_hessian[i][j], self.rtol,
                                       self.atol)
                else:
                    assert hessian[i][j] is None

    def test_create_graph_false(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        assert hessian.stop_gradient == True
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)
        try:
            paddle.grad(hessian, self.x)
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    # TODO(levi): enable this test case when matmul_grad_grad_grad is ok
    def _test_create_graph_true(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = _compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x, create_graph=True)
        assert hessian.stop_gradient == False
        assert np.allclose(hessian.numpy(), numerical_hessian[0][0], self.rtol,
                           self.atol)
        triple_grad = paddle.grad(hessian, self.x)
        assert triple_grad is not None


class TestHessianFloat64(TestHessian):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-5
        self.rtol = 1e-5
        self.atol = 1e-5
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
