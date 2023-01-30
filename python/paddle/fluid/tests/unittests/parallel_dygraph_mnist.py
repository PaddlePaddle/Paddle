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

<<<<<<< HEAD
import numpy as np
from test_dist_base import TestParallelDyGraphRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        pool_size,
        pool_stride,
        pool_padding=0,
        pool_type='max',
        global_pooling=False,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        act=None,
        use_cudnn=False,
        param_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=None,
            bias_attr=None,
        )

        self._pool2d = paddle.nn.MaxPool2D(
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )
=======
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


class SimpleImgConvPool(fluid.dygraph.Layer):

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(num_channels=num_channels,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              stride=conv_stride,
                              padding=conv_padding,
                              dilation=conv_dilation,
                              groups=conv_groups,
                              param_attr=None,
                              bias_attr=None,
                              use_cudnn=use_cudnn)

        self._pool2d = Pool2D(pool_size=pool_size,
                              pool_type=pool_type,
                              pool_stride=pool_stride,
                              pool_padding=pool_padding,
                              global_pooling=global_pooling,
                              use_cudnn=use_cudnn)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu"
        )

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu"
        )

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE)) ** 0.5
        self._fc = paddle.nn.Linear(
            self.pool_2_shape,
            10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale)
            ),
        )
        self.act = paddle.nn.Softmax()
=======

    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(1,
                                                         20,
                                                         5,
                                                         2,
                                                         2,
                                                         act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(20,
                                                         50,
                                                         5,
                                                         2,
                                                         2,
                                                         act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(self.pool_2_shape,
                          10,
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.NormalInitializer(
                                  loc=0.0, scale=scale)),
                          act="softmax")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs, label):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
<<<<<<< HEAD
        x = paddle.reshape(x, shape=[-1, self.pool_2_shape])
        cost = self._fc(x)
        loss = paddle.nn.functional.cross_entropy(
            self.act(cost), label, reduction='none', use_softmax=False
        )
=======
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        cost = self._fc(x)
        loss = fluid.layers.cross_entropy(cost, label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        avg_loss = paddle.mean(loss)
        return avg_loss


class TestMnist(TestParallelDyGraphRunnerBase):
<<<<<<< HEAD
    def get_model(self):
        model = MNIST()
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=2, drop_last=True
        )
        opt = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=model.parameters()
        )
=======

    def get_model(self):
        model = MNIST()
        train_reader = paddle.batch(paddle.dataset.mnist.train(),
                                    batch_size=2,
                                    drop_last=True)
        opt = paddle.optimizer.Adam(learning_rate=1e-3,
                                    parameters=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return model, train_reader, opt

    def run_one_loop(self, model, opt, data):
        batch_size = len(data)
<<<<<<< HEAD
        dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype(
            'float32'
        )
        y_data = (
            np.array([x[1] for x in data])
            .astype('int64')
            .reshape(batch_size, 1)
        )
=======
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data
                           ]).astype('int64').reshape(batch_size, 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True

        avg_loss = model(img, label)

        return avg_loss


if __name__ == "__main__":
    runtime_main(TestMnist)
