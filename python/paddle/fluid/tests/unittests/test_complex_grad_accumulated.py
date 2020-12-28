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
        print(self.A)

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
        self.dtypes = ['float32', 'float64']
        self.theta_size = [4, 4]

    def run_backward(self, device, dtype, mode):
        paddle.set_device(device)

        myLayer = Optimization_ex1(self.theta_size, dtype)

        loss = myLayer(mode)
        loss.backward()

    def test_case_one_step(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.run_backward(dev, dtype, 1)

    def test_case_two_step(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.run_backward(dev, dtype, 2)

    def test_case_non_param(self):
        for dev in self.devices:
            for dtype in self.dtypes:
                self.run_backward(dev, dtype, 3)


if __name__ == '__main__':
    unittest.main()
