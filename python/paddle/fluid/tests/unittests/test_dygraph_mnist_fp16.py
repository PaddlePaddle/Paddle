# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.framework import _test_eager_guard


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
                 dtype='float32',
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
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            dtype=dtype,
            act=act)

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


class MNIST(fluid.dygraph.Layer):
    def __init__(self, dtype="float32"):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            num_channels=3,
            num_filters=20,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu",
            dtype=dtype,
            use_cudnn=True)

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            num_channels=20,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu",
            dtype=dtype,
            use_cudnn=True)

        self.pool_2_shape = 50 * 53 * 53
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._linear = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax",
            dtype=dtype)

    def forward(self, inputs, label):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        cost = self._linear(x)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        return avg_loss


class TestMnist(unittest.TestCase):
    def func_mnist_fp16(self):
        if not fluid.is_compiled_with_cuda():
            return
        x = np.random.randn(1, 3, 224, 224).astype("float16")
        y = np.random.randint(10, size=[1, 1], dtype="int64")
        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            model = MNIST(dtype="float16")
            x = fluid.dygraph.to_variable(x)
            y = fluid.dygraph.to_variable(y)
            loss = model(x, y)
            print(loss.numpy())

    def test_mnist_fp16(self):
        with _test_eager_guard():
            self.func_mnist_fp16()
        self.func_mnist_fp16()


if __name__ == "__main__":
    unittest.main()
