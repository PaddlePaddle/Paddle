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

from __future__ import print_function

import unittest
import numpy as np

import paddle

import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class Optimization_ex1(paddle.nn.Layer):
    def __init__(self,
                 shape,
                 dtype,
                 param_attr=paddle.nn.initializer.Uniform(
                     low=-5., high=5.)):
        super(Optimization_ex1, self).__init__()

        self.theta0 = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.theta1 = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(
            np.random.random((4, 4)).astype(dtype) + np.random.random((4, 4))
            .astype(dtype) * 1j)
        self.B = paddle.to_tensor(
            np.random.random((4, 4)).astype(dtype) + np.random.random(
                (4, 4)).astype(dtype) * 1j,
            stop_gradient=False)

    def forward(self, mode=1):
        jj = paddle.to_tensor(np.array([1j]).astype(np.complex64))
        if mode == 1:
            # run all calc in one step
            loss = paddle.sum(self.A + (self.theta0 + self.theta1 * jj)) * (
                paddle.sum(self.A + (self.theta0 + self.theta1 * jj)).conj())
            return loss.real()
        elif mode == 2:
            # run in two step
            self.theta = self.theta0 + self.theta1 * jj
            loss = paddle.sum(self.A + self.theta) * (
                paddle.sum(self.A + self.theta).conj())
            return loss.real()
        elif mode == 3:
            # run without param
            loss = paddle.sum(self.A + self.B) * (
                paddle.sum(self.A + self.B).conj())
            return loss.real()
        else:
            raise NotImplementedError


class TestComplexGradAccumulated(unittest.TestCase):
    def setUp(self):
        self.devices = ['cpu']
        if core.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.iter = 3
        self.learning_rate = 0.5
        self.dtypes = ['float32', 'float64']
        self.theta_size = [4, 4]

    def train(self, device, dtype, mode):
        paddle.set_device(device)

        myLayer = Optimization_ex1(self.theta_size, dtype)
        optimizer = paddle.optimizer.SGD(learning_rate=self.learning_rate,
                                         parameters=myLayer.parameters())

        for iter in range(self.iter):
            loss = myLayer(mode)
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

    def train_no_clear_grad(self, device, dtype, mode):
        paddle.set_device(device)

        myLayer = Optimization_ex1(self.theta_size, dtype)
        optimizer = paddle.optimizer.SGD(learning_rate=self.learning_rate,
                                         parameters=myLayer.parameters())

        for iter in range(self.iter):
            loss = myLayer(mode)
            loss.backward()

            optimizer.step()

    def test_case_one_step(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 1)
                self.train_no_clear_grad(dev, dtype, 1)

    def test_case_two_step(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 2)
                self.train_no_clear_grad(dev, dtype, 2)

    def test_case_non_param(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 3)
                self.train_no_clear_grad(dev, dtype, 3)

    def test_eager(self):
        with _test_eager_guard():
            self.test_case_one_step()
            self.test_case_two_step()
            self.test_case_non_param()


if __name__ == '__main__':
    unittest.main()
