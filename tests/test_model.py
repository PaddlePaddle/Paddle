# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import unittest

import os

import sys
sys.path.append('../')

import numpy as np
import contextlib

import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from model import Model, CrossEntropy, Input, Loss, init_context
from metrics import Accuracy
from callbacks import ProgBarLogger
from paddle.fluid.io import BatchSampler, DataLoader
from paddle.fluid.io import MNIST as MnistDataset


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
                 conv_groups=None,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__('SimpleConv')

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


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()
        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            800,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, [-1, 1, 28, 28])
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self._fc(x)
        return x


class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        SIZE = 10
        self._fc1 = Linear(784, 200, act="relu")
        self._fc2 = Linear(200, 200, act="relu")
        self._fc3 = Linear(200, 200, act="relu")
        self._fc4 = Linear(200, 10, act="softmax")
        self._fc5 = Linear(200, 10, act="softmax")

    def forward(self, inputs):
        x1 = self._fc1(inputs)
        x2 = self._fc2(x1)
        x3 = self._fc3(x2)
        o1 = self._fc5(x3)
        o2 = self._fc4(x2)
        return o1, o2


class MyCrossEntropy(Loss):
    def __init__(self, average=True):
        super(MyCrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        loss1 = fluid.layers.cross_entropy(outputs[0], labels[0])
        loss2 = fluid.layers.cross_entropy(outputs[1], labels[0])
        return [loss1, loss2]


class TestModel(unittest.TestCase):
    def fit(self, dynamic, is_mlp=False):
        init_context('dynamic' if dynamic else 'static')

        im_shape = (-1, 784)
        batch_size = 128

        inputs = [Input(im_shape, 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        train_dataset = MnistDataset(mode='train')
        val_dataset = MnistDataset(mode='test')

        model = MNIST() if not is_mlp else MLP()
        optim = fluid.optimizer.Momentum(
            learning_rate=0.01, momentum=.9, parameter_list=model.parameters())
        loss = CrossEntropy() if not is_mlp else MyCrossEntropy()
        model.prepare(optim, loss, Accuracy(), inputs, labels)
        cbk = ProgBarLogger(50)
        model.fit(train_dataset,
                  val_dataset,
                  epochs=2,
                  batch_size=batch_size,
                  callbacks=cbk)

    def test_fit_static(self):
        self.fit(False)

    def test_fit_dygraph(self):
        self.fit(True)

    def test_fit_static_multi_loss(self):
        self.fit(False, MyCrossEntropy())

    def test_fit_dygraph_multi_loss(self):
        self.fit(True, MyCrossEntropy())


if __name__ == '__main__':
    unittest.main()
