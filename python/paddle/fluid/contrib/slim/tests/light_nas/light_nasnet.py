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
"""LightNASNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ['LightNASNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class LightNASNet(object):
    """LightNASNet."""

    def __init__(self):
        self.params = train_parameters

    def net(self, input, bottleneck_params_list=None, class_dim=1000,
            scale=1.0):
        """Build network.
        Args:
            input: Variable, input.
            class_dim: int, class dim.
            scale: float, scale.
        Returns:
            Variable, network output.
        """
        if bottleneck_params_list is None:
            # MobileNetV2
            # bottleneck_params_list = [
            #     (1, 16, 1, 1, 3, 1, 0),
            #     (6, 24, 2, 2, 3, 1, 0),
            #     (6, 32, 3, 2, 3, 1, 0),
            #     (6, 64, 4, 2, 3, 1, 0),
            #     (6, 96, 3, 1, 3, 1, 0),
            #     (6, 160, 3, 2, 3, 1, 0),
            #     (6, 320, 1, 1, 3, 1, 0),
            # ]
            bottleneck_params_list = [
                (1, 16, 1, 1, 3, 1, 0),
                (3, 24, 3, 2, 3, 1, 0),
                (3, 40, 3, 2, 5, 1, 0),
                (6, 80, 3, 2, 5, 1, 0),
                (6, 96, 2, 1, 3, 1, 0),
                (6, 192, 4, 2, 5, 1, 0),
                (6, 320, 1, 1, 3, 1, 0),
            ]

        #conv1
        input = self.conv_bn_layer(
            input,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            name='conv1_1')

        # bottleneck sequences
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s, k, ifshortcut, ifse = layer_setting
            i += 1
            input = self.invresi_blocks(
                input=input,
                in_channel=in_c,
                expansion=t,
                out_channel=int(c * scale),
                num_layers=n,
                stride=s,
                filter_size=k,
                shortcut=ifshortcut,
                squeeze=ifse,
                name='conv' + str(i))
            in_c = int(c * scale)
        #last_conv
        input = self.conv_bn_layer(
            input=input,
            num_filters=int(1280 * scale) if scale > 1.0 else 1280,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            name='conv9')

        input = fluid.layers.pool2d(
            input=input,
            pool_size=7,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)

        output = fluid.layers.fc(input=input,
                                 size=class_dim,
                                 param_attr=ParamAttr(name='fc10_weights'),
                                 bias_attr=ParamAttr(name='fc10_offset'))
        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      name=None,
                      use_cudnn=True):
        """Build convolution and batch normalization layers.
        Args:
            input: Variable, input.
            filter_size: int, filter size.
            num_filters: int, number of filters.
            stride: int, stride.
            padding: int, padding.
            num_groups: int, number of groups.
            if_act: bool, whether using activation.
            name: str, name.
            use_cudnn: bool, whether use cudnn.
        Returns:
            Variable, layers output.
        """
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            return fluid.layers.relu6(bn)
        else:
            return bn

    def shortcut(self, input, data_residual):
        """Build shortcut layer.
        Args:
            input: Variable, input.
            data_residual: Variable, residual layer.
        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        """Build squeeze excitation layers.
        Args:
            input: Variable, input.
            num_channels: int, number of channels.
            reduction_ratio: float, reduction ratio.
            name: str, name.
        Returns:
            Variable, layers output.
        """
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def inverted_residual_unit(self,
                               input,
                               num_in_filter,
                               num_filters,
                               ifshortcut,
                               ifse,
                               stride,
                               filter_size,
                               expansion_factor,
                               reduction_ratio=4,
                               name=None):
        """Build inverted residual unit.
        Args:
            input(Variable): Theinput.
            num_in_filter(int): The number of input filters.
            num_filters(int): The number of filters.
            ifshortcut(bool): Whether to use shortcut.
            stride(int): The stride.
            filter_size(int): The filter size.
            padding(int): The padding.
            expansion_factor(float): Expansion factor.
            name(str): The name.
        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))
        channel_expand = self.conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name=name + '_expand')

        bottleneck_conv = self.conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) / 2),
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = self.conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')
        out = linear_out
        if ifshortcut:
            out = self.shortcut(input=input, data_residual=out)
        if ifse:
            scale = self.squeeze_excitation(
                input=linear_out,
                num_channels=num_filters,
                reduction_ratio=reduction_ratio,
                name=name + '_fc')
            out = fluid.layers.elementwise_add(x=out, y=scale, act='relu')
        return out

    def invresi_blocks(self,
                       input,
                       in_channel,
                       expansion,
                       out_channel,
                       num_layers,
                       stride,
                       filter_size,
                       shortcut,
                       squeeze,
                       name=None):
        """Build inverted residual blocks.
        Args:
            input(Variable): The input feture map.
            in_channel(int): The number of input channel.
            expansion(float): Expansion factor.
            out_channel(int): The number of output channel.
            num_layers(int): The number of layers.
            stride(int): The stride.
            filter_size(int): The size of filter.
            shortcut(bool): Whether to add shortcut layers.
            squeeze(bool): Whether to add squeeze excitation layers.
            name(str): The name.
        Returns:
            Variable, layers output.
        """
        first_block = self.inverted_residual_unit(
            input=input,
            num_in_filter=in_channel,
            num_filters=out_channel,
            ifshortcut=False,
            ifse=squeeze,
            stride=stride,
            filter_size=filter_size,
            expansion_factor=expansion,
            name=name + '_1')

        last_residual_block = first_block
        last_c = out_channel

        for i in range(1, num_layers):
            last_residual_block = self.inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=out_channel,
                ifshortcut=shortcut,
                ifse=squeeze,
                stride=1,
                filter_size=filter_size,
                expansion_factor=expansion,
                name=name + '_' + str(i + 1))
        return last_residual_block
