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

import contextlib
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.imperative.nn import Conv2D, Pool2D, FC
from paddle.fluid.imperative.base import to_variable


class SimpleImgConvPool(fluid.imperative.PyLayer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
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

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.imperative.PyLayer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MNIST, self).__init__(param_attr=param_attr, bias_attr=bias_attr)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 5, 20, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 5, 50, 2, 2, act="relu")

        pool_2_shape = 50 * 8 * 8
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(-1,
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)))

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        return x


class TestImperativeMnist(unittest.TestCase):
    def test_mnist_cpu_float32(self):
        with fluid.imperative.guard():
            mnist = MNIST()
            sgd = SGDOptimizer(learning_rate=1e-3)

            for i in range(2):
                x_data = np.random.rand(128, 1, 28, 28).astype('float32')
                img = to_variable(x_data)
                y_data = np.random.rand(128, 1).astype('int64')
                label = to_variable(y_data)
                label._stop_gradient = True

                predict = mnist(img)
                out = fluid.layers.cross_entropy(predict, label)
                out._backward()
                sgd.minimize(out)


if __name__ == '__main__':
    unittest.main()
