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

from __future__ import division
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, BatchNorm

# cudnn is not better when batch size is 1.
use_cudnn = False
import numpy as np


class ConvBN(fluid.dygraph.Layer):
    """docstring for Conv2D"""

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 stddev=0.02,
                 norm=True,
                 is_test=False,
                 act='leaky_relu',
                 relufactor=0.0,
                 use_bias=False):
        super(ConvBN, self).__init__()

        pattr = fluid.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=pattr,
            bias_attr=use_bias)
        if norm:
            self.bn = BatchNorm(
                num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,
                                                                    0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)),
                is_test=False,
                trainable_statistics=True)
        self.relufactor = relufactor
        self.norm = norm
        self.act = act

    def forward(self, inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)

        if self.act == 'leaky_relu':
            conv = fluid.layers.leaky_relu(conv, alpha=self.relufactor)
        elif self.act == 'relu':
            conv = fluid.layers.relu(conv)
        else:
            conv = conv

        return conv


class DeConvBN(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=[0, 0],
                 outpadding=[0, 0, 0, 0],
                 stddev=0.02,
                 act='leaky_relu',
                 norm=True,
                 is_test=False,
                 relufactor=0.0,
                 use_bias=False):
        super(DeConvBN, self).__init__()

        pattr = fluid.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        self._deconv = Conv2DTranspose(
            num_channels,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            param_attr=pattr,
            bias_attr=use_bias)
        if norm:
            self.bn = BatchNorm(
                num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,
                                                                    0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)),
                is_test=False,
                trainable_statistics=True)
        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.act = act

    def forward(self, inputs):
        conv = self._deconv(inputs)
        conv = fluid.layers.pad2d(
            conv, paddings=self.outpadding, mode='constant', pad_value=0.0)

        if self.norm:
            conv = self.bn(conv)

        if self.act == 'leaky_relu':
            conv = fluid.layers.leaky_relu(conv, alpha=self.relufactor)
        elif self.act == 'relu':
            conv = fluid.layers.relu(conv)
        else:
            conv = conv

        return conv
