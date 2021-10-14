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
import paddle.nn.functional as F
from utils import _compute_numerical_vhp


class TestVHP(unittest.TestCase):
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
        self.vx = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vy = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_func_output = func(self.x).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, self.x, self.vx, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, self.x, self.vx)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)

    def test_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_func_output = func(self.x, self.y).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, [self.x, self.y], [self.vx, self.vy], self.numerical_delta,
            self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y],
                                               [self.vx, self.vy])
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        for i in range(len(vhp)):
            assert np.allclose(vhp[i].numpy(), numerical_vhp[i], self.rtol,
                               self.atol)

    def test_v_default(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_func_output = func(self.x, self.y).numpy()
        vx = paddle.ones(self.vx.shape, dtype=self.vx.dtype)
        vy = paddle.ones(self.vy.shape, dtype=self.vy.dtype)
        numerical_vhp = _compute_numerical_vhp(func, [self.x, self.y],
                                               [vx, vy], self.numerical_delta,
                                               self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y])
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        for i in range(len(vhp)):
            assert np.allclose(vhp[i].numpy(), numerical_vhp[i], self.rtol,
                               self.atol)

    def test_allow_unused_false(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        try:
            self.x.stop_gradient = False
            self.y.stop_gradient = False
            _ = paddle.autograd.vhp(func, [self.x, self.y])
        except ValueError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("allow_unused") > 0

    def test_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_func_output = func(self.x, self.y).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, [self.x, self.y], [self.vx, self.vy], self.numerical_delta,
            self.np_dtype)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, [self.x, self.y],
                                               [self.vx, self.vy],
                                               allow_unused=True)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)
        assert vhp[1] is None

    def test_create_graph_false(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_func_output = func(self.x).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, self.x, self.vx, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func, self.x, self.vx)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert vhp[0].stop_gradient == True
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)
        try:
            paddle.grad(vhp, self.x)
        except RuntimeError as e:
            error_msg = cpt.get_exception_message(e)
            assert error_msg.find("has no gradient") > 0

    def test_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_func_output = func(self.x).numpy()
        numerical_vhp = _compute_numerical_vhp(
            func, self.x, self.vx, self.numerical_delta, self.np_dtype)

        self.x.stop_gradient = False
        func_output, vhp = paddle.autograd.vhp(func,
                                               self.x,
                                               self.vx,
                                               create_graph=True)
        assert np.allclose(func_output.numpy(), numerical_func_output,
                           self.rtol, self.atol)
        assert vhp[0].stop_gradient == False
        assert np.allclose(vhp[0].numpy(), numerical_vhp[0], self.rtol,
                           self.atol)
        triple_grad = paddle.grad(vhp, self.x)
        assert triple_grad is not None


class TestVHPFloat64(TestVHP):
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
        self.vx = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.vy = paddle.rand(shape=self.shape, dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
