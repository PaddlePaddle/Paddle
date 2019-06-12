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
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.layer_helper import LayerHelper

from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None)

        self._batch_norm = BatchNorm(
            self.full_name(), num_filters, act=act, momentum=0.1)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class SqueezeExcitation(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, reduction_ratio):

        super(SqueezeExcitation, self).__init__(name_scope)
        self._pool = Pool2D(
            self.full_name(), pool_size=0, pool_type='avg', global_pooling=True)
        self._squeeze = FC(
            self.full_name(),
            size=num_channels // reduction_ratio,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.05)),
            act='relu')
        self._excitation = FC(
            self.full_name(),
            size=num_channels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.05)),
            act='sigmoid')

    def forward(self, input):
        y = self._pool(input)
        y = self._squeeze(y)
        y = self._excitation(y)
        y = fluid.layers.elementwise_mul(x=input, y=y, axis=0)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 reduction_ratio,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1)
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality)
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act='relu')

        self.scale = SqueezeExcitation(
            self.full_name(),
            num_channels=num_filters * 4,
            reduction_ratio=reduction_ratio)

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=scale)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        y = layer_helper.append_activation(y)
        return y


class SeResNeXt(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=102):
        super(SeResNeXt, self).__init__(name_scope)

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                self.full_name(),
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_channels=3,
                num_filters=3,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                self.full_name(),
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_channels=3,
                num_filters=3,
                filter_size=7,
                stride=2,
                act='relu')
            self.conv1 = ConvBNLayer(
                self.full_name(),
                num_channels=64,
                num_filters=3,
                filter_size=7,
                stride=2,
                act='relu')
            self.conv2 = ConvBNLayer(
                self.full_name(),
                num_channels=64,
                num_filters=3,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                self.full_name(),
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            self.full_name(), pool_size=7, pool_type='avg', global_pooling=True)
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.fc = FC(self.full_name(),
                     size=class_dim,
                     act='softmax',
                     param_attr=fluid.param_attr.ParamAttr(
                         initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label):
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            y = self.pool(y)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(inputs)
            y = self.conv2(inputs)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.dropout(y, dropout_prob=0.2, seed=1)
        cost = self.fc(y)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        return avg_loss


class TestSeResNeXt(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SeResNeXt("se-resnext")
        train_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False),
            batch_size=2,
            drop_last=True)

        opt = fluid.optimizer.SGD(learning_rate=1e-3)
        return model, train_reader, opt

    def run_one_loop(self, model, opt, data):
        bs = len(data)
        dy_x_data = np.array([x[0].reshape(3, 224, 224)
                              for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(bs, 1)
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True

        loss = model(img, label)
        return loss


if __name__ == "__main__":
    runtime_main(TestSeResNeXt)
