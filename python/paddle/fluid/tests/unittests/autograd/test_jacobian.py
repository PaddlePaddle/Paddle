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
from paddle.autograd.functional import _check_tensors


def _product(t):
    if isinstance(t, int):
        return t
    else:
        return np.product(t)


def _get_item(t, idx):
    assert isinstance(t, paddle.Tensor), "The first argument t must be Tensor."
    assert isinstance(idx,
                      int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    return flat_t.__getitem__(idx)


def _set_item(t, idx, value):
    assert isinstance(t, paddle.Tensor), "The first argument t must be Tensor."
    assert isinstance(idx,
                      int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    flat_t.__setitem__(idx, value)
    return paddle.reshape(flat_t, t.shape)


def _compute_numerical_jacobian(func, xs, delta, np_dtype):
    xs = _check_tensors(xs, "xs")
    ys = _check_tensors(func(*xs), "ys")
    fin_size = len(xs)
    fout_size = len(ys)
    jacobian = list([] for _ in range(fout_size))
    for i in range(fout_size):
        jac_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            jac_i[j] = np.zeros(
                (_product(ys[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        jacobian[i] = jac_i

    for j in range(fin_size):
        for q in range(_product(xs[j].shape)):
            orig = _get_item(xs[j], q)
            x_pos = orig + delta
            xs[j] = _set_item(xs[j], q, x_pos)
            ys_pos = _check_tensors(func(*xs), "ys_pos")

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = _check_tensors(func(*xs), "ys_neg")

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.
    return jacobian


class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 4)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-5
        self.rtol = 1e-3
        self.atol = 1e-2
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
        self.atol = 1e-5
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    # NOTE(levi): skip this test case temporaryly.
    def test_create_graph_true(self):
        pass


if __name__ == "__main__":
    unittest.main()
