#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
                 param_attr=paddle.nn.initializer.Uniform(
                     low=-5., high=5.),
                 dtype='float32'):
        super(Optimization_ex1, self).__init__()

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(
            np.random.randn(4, 4) + np.random.randn(4, 4) * 1j)

    def forward(self):
        loss = paddle.add(self.theta, self.A)
        return loss.real()


class TestComplexSimpleNet(unittest.TestCase):
    def setUp(self):
        self.devices = ['cpu']
        if core.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.iter = 10
        self.learning_rate = 0.5
        self.theta_size = [4, 4]

    def train(self, device):
        paddle.set_device(device)

        myLayer = Optimization_ex1(self.theta_size)
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, parameters=myLayer.parameters())

        for itr in range(self.iter):
            loss = myLayer()
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

    def test_train_success(self):
        for dev in self.devices:
            self.train(dev)

    def test_eager(self):
        with _test_eager_guard():
            self.test_train_success()


if __name__ == '__main__':
    unittest.main()
