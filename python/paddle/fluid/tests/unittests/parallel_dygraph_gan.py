# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import contextlib
import unittest
import numpy as np
import six
import pickle

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable

from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase
from parallel_dygraph_mnist import SimpleImgConvPool


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        SIZE = 10

        # A net
        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")
        self.pool_2_shape = 50 * 4 * 4

        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc_1 = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

        # B net
        self._simple_img_conv_pool_3 = SimpleImgConvPool(
            20, 30, 5, 2, 2, act="relu")
        self.pool_3_shape = 30 * 4 * 4
        scale = (2.0 / (self.pool_3_shape**2 * SIZE))**0.5
        self._fc_2 = Linear(
            self.pool_3_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")
        self.step = 0

    def forward(self, inputs, label):
        x = self._simple_img_conv_pool_1(inputs)

        if self.step % 2 == 0:
            x = self._simple_img_conv_pool_2(x)
            x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
            cost = self._fc_1(x)
        else:
            x = self._simple_img_conv_pool_3(x)
            x.stop_gradient = True
            x = fluid.layers.reshape(x, shape=[-1, self.pool_3_shape])
            cost = self._fc_2(x)

        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        self.step += 1
        return avg_loss


class TestGanNet(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = MNIST()
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=2, drop_last=True)
        opt = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=model.parameters())
        return model, train_reader, opt

    def run_one_loop(self, model, opt, data):
        batch_size = len(data)
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True

        avg_loss = model(img, label)

        return avg_loss


if __name__ == "__main__":
    runtime_main(TestGanNet)
