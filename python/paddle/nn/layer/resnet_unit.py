#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import collections
import itertools
import six
import math
import sys
import warnings
from functools import partial, reduce

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle import framework
from paddle.device import get_device, get_cudnn_version
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.nn import Layer, LayerList
from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.data_feeder import convert_dtype
from paddle import _C_ops
__all__ = []


class ResNetUnit(Layer):
    r"""
    ******Temporary version******.
    ResNetUnit is designed for optimize the performence by using cudnnv8 API.
    """

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 ele_count=1,
                 momentum=0.9,
                 eps=1e-5,
                 conv_format='NHWC',
                 bn_format='NHWC',
                 act='relu',
                 fused_add=False,
                 has_shortcut=False,
                 filter_x_attr=None,
                 scale_x_attr=None,
                 bias_x_attr=None,
                 filter_z_attr=None,
                 scale_z_attr=None,
                 bias_z_attr=None,
                 name=None):
        super(ResNetUnit, self).__init__()
        self._in_channels = num_channels
        self._out_channels = num_filters
        self._stride = stride
        self._dilation = 1
        self._kernel_size = utils.convert_to_list(filter_size, 2, 'kernel_size')
        self._padding = (filter_size - 1) // 2
        self._groups = 1
        self._ele_count = ele_count
        self._momentum = momentum
        self._eps = eps
        self._conv_format = conv_format
        self._bn_format = bn_format
        self._act = act
        self._fused_add = fused_add
        self._has_shortcut = has_shortcut
        self._filter_x_attr = filter_x_attr
        self._scale_x_attr = scale_x_attr
        self._bias_x_attr = bias_x_attr
        self._filter_z_attr = filter_z_attr
        self._scale_z_attr = scale_z_attr
        self._bias_z_attr = bias_z_attr

        # check format
        valid_format = {'NHWC'}
        if conv_format not in valid_format:
            raise ValueError(
                "conv_format must be one of {}, but got conv_format='{}'".
                format(valid_format, conv_format))
        if bn_format not in valid_format:
            raise ValueError(
                "bn_format must be one of {}, but got bn_format='{}'".format(
                    valid_format, bn_format))

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return I.Normal(0.0, std)

        # initial filter
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32
        bn_param_shape = [1, 1, 1, num_filters]
        filter_shape = [num_filters, filter_size, filter_size, num_channels]
        self.filter_x = self.create_parameter(
            shape=filter_shape,
            attr=self._filter_x_attr,
            default_initializer=_get_default_param_initializer()).astype(
                np.float16)
        self.scale_x = self.create_parameter(
            shape=bn_param_shape,
            attr=self._scale_x_attr,
            dtype=bn_param_dtype,
            default_initializer=I.Constant(1.0))
        self.bias_x = self.create_parameter(
            shape=bn_param_shape,
            attr=self._bias_x_attr,
            dtype=bn_param_dtype,
            is_bias=True)
        if has_shortcut:
            self.filter_z = self.create_parameter(
                shape=filter_shape,
                attr=self._filter_z_attr,
                default_initializer=_get_default_param_initializer()).astype(
                    np.float16)
            self.scale_z = self.create_parameter(
                shape=bn_param_shape,
                attr=self._scale_z_attr,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(1.0))
            self.bias_z = self.create_parameter(
                shape=bn_param_shape,
                attr=self._bias_z_attr,
                dtype=bn_param_dtype,
                is_bias=True)
        else:
            self.filter_z = self.filter_x
            self.scale_z = self.scale_x
            self.bias_z = self.bias_x

    def forward(self, x, z=None):
        if self._fused_add and z is None:
            raise ValueError("z can not be None")
        if self._fused_add is False:
            z = x

        out = F.resnet_unit(x, self.filter_x, self.scale_x, self.bias_x, z,
                            self.filter_z, self.scale_z, self.bias_z,
                            self._ele_count, self._stride, self._padding,
                            self._dilation, self._groups, self._momentum,
                            self._eps, self._conv_format, self._bn_format,
                            self._fused_add, self._has_shortcut, self._act)
        return out
