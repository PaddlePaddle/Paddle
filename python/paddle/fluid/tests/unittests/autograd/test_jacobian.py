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
from utils import _compute_numerical_jacobian, _compute_numerical_batch_jacobian


class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 4)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input_and_single_output(self):
        def func(x):
            return paddle.matmul(x, x)

        numerical_jacobian = _compute_numerical_jacobian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        jacobian = paddle.autograd.jacobian(func, self.x)
        assert np.allclose(jacobian.numpy(), numerical_jacobian[0][0],
                           self.rtol, self.atol)

    def test_single_input_and_multi_output(self):
        def func(x):
            return paddle.matmul(x, x), x * x

        numerical_jacobian = _compute_numerical_jacobian(
            func, self.x, self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        jacobian = paddle.autograd.jacobian(func, self.x)
        for i in range(len(jacobian)):
            assert np.allclose(jacobian[i].numpy(), numerical_jacobian[i][0],
                               self.rtol, self.atol)

    def test_multi_input_and_single_output(self):
        def func(x, y):
            return paddle.matmul(x, y)

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
        for j in range(len(jacobian)):
            assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
                               self.rtol, self.atol)

    def test_multi_input_and_multi_output(self):
        def func(x, y):
            return paddle.matmul(x, y), x * y

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
        for i in range(len(jacobian)):
            for j in range(len(jacobian[0])):
                assert np.allclose(jacobian[i][j].numpy(),
                                   numerical_jacobian[i][j], self.rtol,
                                   self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return paddle.matmul(x, x)

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.matmul(x, x)

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.jacobian(
            func, [self.x, self.y], allow_unused=True)
        assert np.allclose(jacobian[0].numpy(), numerical_jacobian[0][0],
                           self.rtol, self.atol)
        assert jacobian[1] is None

    def test_create_graph_false(self):
        def func(x, y):
            return paddle.matmul(x, y)

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
        for j in range(len(jacobian)):
            assert jacobian[j].stop_gradient == True
            assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
                               self.rtol, self.atol)
        try:
            paddle.grad(jacobian[0], [self.x, self.y])
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    def test_create_graph_true(self):
        def func(x, y):
            return paddle.matmul(x, y)

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.jacobian(
            func, [self.x, self.y], create_graph=True)
        for j in range(len(jacobian)):
            assert jacobian[j].stop_gradient == False
            assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
                               self.rtol, self.atol)
        double_grad = paddle.grad(jacobian[0], [self.x, self.y])
        assert double_grad is not None


class TestJacobianFloat64(TestJacobian):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 4)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-7
        self.rtol = 1e-7
        self.atol = 1e-7
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)


class TestJacobianBatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.x_shape = (4, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (4, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)

    def test_batch_single_input_and_batch_single_output(self):
        def func(x):
            return paddle.matmul(paddle.matmul(x, self.weight), self.y)

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        batch_jacobian = paddle.autograd.batch_jacobian(
            func,
            self.x, )

        self.assertTrue(
            np.allclose(batch_jacobian.numpy().all(), numerical_jacobian[0][0]
                        .all()))

    def test_batch_single_input_and_batch_multi_output(self):
        def func(x):
            return paddle.matmul(paddle.matmul(x, self.weight), self.y), x * x

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        batch_jacobian = paddle.autograd.batch_jacobian(
            func,
            self.x, )

        for i in range(len(batch_jacobian)):
            assert np.allclose(batch_jacobian[i].numpy(),
                               numerical_jacobian[i][0], self.rtol, self.atol)

    def test_batch_multi_input_and_batch_single_output(self):
        def func(x, y):
            return x * y

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        batch_jacobian = paddle.autograd.batch_jacobian(func, [self.x, self.y])

        for j in range(len(batch_jacobian)):
            assert np.allclose(batch_jacobian[j].numpy(),
                               numerical_jacobian[0][j], self.rtol, self.atol)

    def test_batch_multi_input_and_batch_multi_output(self):
        def func(x, y):
            return x * y, x * y

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        batch_jacobian = paddle.autograd.batch_jacobian(func, [self.x, self.y])

        for i in range(len(batch_jacobian)):
            assert np.allclose(batch_jacobian[i], numerical_jacobian[i],
                               self.rtol, self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return x * x

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            jacobian = paddle.autograd.batch_jacobian(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return x * x

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.batch_jacobian(
            func, [self.x, self.y], allow_unused=True)

        assert np.allclose(jacobian[0].numpy(), numerical_jacobian[0][0],
                           self.rtol, self.atol)
        assert jacobian[1] is None

    def test_create_graph_false(self):
        def func(x, y):
            return x * y

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.batch_jacobian(func, [self.x, self.y])
        for j in range(len(jacobian)):
            assert jacobian[j].stop_gradient == True
            assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
                               self.rtol, self.atol)
        try:
            paddle.grad(jacobian[0], [self.x, self.y])
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    def test_create_graph_true(self):
        def func(x, y):
            return x * y

        numerical_jacobian = _compute_numerical_batch_jacobian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        jacobian = paddle.autograd.batch_jacobian(
            func, [self.x, self.y], create_graph=True)
        for j in range(len(jacobian)):
            assert jacobian[j].stop_gradient == False
            assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
                               self.rtol, self.atol)
        double_grad = paddle.grad(jacobian[0], [self.x, self.y])
        assert double_grad is not None


class TestJacobianBatchFloat64(TestJacobianBatch):
    @classmethod
    def setUpClass(self):
        self.x_shape = (12, 2)
        self.weight_shape = (2, 12)
        self.y_shape = (12, 2)
        self.dtype = 'float64'
        self.np_dtype = np.float64
        self.numerical_delta = 1e-7
        self.rtol = 1e-7
        self.atol = 1e-7
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
