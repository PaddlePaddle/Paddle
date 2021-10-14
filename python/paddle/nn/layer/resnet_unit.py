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
from paddle.fluid.param_attr import ParamAttr
from paddle import _C_ops
__all__ = []


class ResNetUnit(Layer):
    r"""
    ******Temporary version******.
    ResNetUnit is designed for optimize the performence by using cudnnv8 API.
    """

    def __init__(self,
                 num_channels_x,
                 num_filters,
                 filter_size,
                 stride=1,
                 momentum=0.9,
                 eps=1e-5,
                 data_format='NHWC',
                 act='relu',
                 fuse_add=False,
                 has_shortcut=False,
                 use_global_stats=False,
                 is_test=False,
                 filter_x_attr=None,
                 scale_x_attr=None,
                 bias_x_attr=None,
                 moving_mean_x_name=None,
                 moving_var_x_name=None,
                 num_channels_z=1,
                 stride_z=1,
                 filter_z_attr=None,
                 scale_z_attr=None,
                 bias_z_attr=None,
                 moving_mean_z_name=None,
                 moving_var_z_name=None):
        super(ResNetUnit, self).__init__()
        self._stride = stride
        self._stride_z = stride_z
        self._dilation = 1
        self._kernel_size = utils.convert_to_list(filter_size, 2, 'kernel_size')
        self._padding = (filter_size - 1) // 2
        self._groups = 1
        self._momentum = momentum
        self._eps = eps
        self._data_format = data_format
        self._act = act
        self._fuse_add = fuse_add
        self._has_shortcut = has_shortcut
        self._use_global_stats = use_global_stats
        self._is_test = is_test

        # check format
        valid_format = {'NHWC'}
        if data_format not in valid_format:
            raise ValueError(
                "conv_format must be one of {}, but got conv_format='{}'".
                format(valid_format, data_format))

        def _get_default_param_initializer(channels):
            filter_elem_num = np.prod(self._kernel_size) * channels
            std = (2.0 / filter_elem_num)**0.5
            return I.Normal(0.0, std)

        # initial filter
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32
        bn_param_shape = [1, 1, 1, num_filters]
        filter_x_shape = [num_filters, filter_size, filter_size, num_channels_x]
        filter_z_shape = [num_filters, filter_size, filter_size, num_channels_z]

        self.filter_x = self.create_parameter(
            shape=filter_x_shape,
            attr=filter_x_attr,
            default_initializer=_get_default_param_initializer(num_channels_x))
        self.scale_x = self.create_parameter(
            shape=bn_param_shape,
            attr=scale_x_attr,
            dtype=bn_param_dtype,
            default_initializer=I.Constant(1.0))
        self.bias_x = self.create_parameter(
            shape=bn_param_shape,
            attr=bias_x_attr,
            dtype=bn_param_dtype,
            is_bias=True)
        self.mean_x = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_x_name,
                initializer=I.Constant(0.0),
                trainable=False),
            shape=bn_param_shape,
            dtype=bn_param_dtype)
        self.mean_x.stop_gradient = True
        self.var_x = self.create_parameter(
            attr=ParamAttr(
                name=moving_var_x_name,
                initializer=I.Constant(1.0),
                trainable=False),
            shape=bn_param_shape,
            dtype=bn_param_dtype)
        self.var_x.stop_gradient = True
        if has_shortcut:
            self.filter_z = self.create_parameter(
                shape=filter_z_shape,
                attr=filter_z_attr,
                default_initializer=_get_default_param_initializer(
                    num_channels_z))
            self.scale_z = self.create_parameter(
                shape=bn_param_shape,
                attr=scale_z_attr,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(1.0))
            self.bias_z = self.create_parameter(
                shape=bn_param_shape,
                attr=bias_z_attr,
                dtype=bn_param_dtype,
                is_bias=True)
            self.mean_z = self.create_parameter(
                attr=ParamAttr(
                    name=moving_mean_z_name,
                    initializer=I.Constant(0.0),
                    trainable=False),
                shape=bn_param_shape,
                dtype=bn_param_dtype)
            self.mean_z.stop_gradient = True
            self.var_z = self.create_parameter(
                attr=ParamAttr(
                    name=moving_var_z_name,
                    initializer=I.Constant(1.0),
                    trainable=False),
                shape=bn_param_shape,
                dtype=bn_param_dtype)
            self.var_z.stop_gradient = True
        else:
            self.filter_z = None
            self.scale_z = None
            self.bias_z = None
            self.mean_z = None
            self.var_z = None

    def forward(self, x, z=None):
        if self._fuse_add and z is None:
            raise ValueError("z can not be None")

        out = F.resnet_unit(
            x, self.filter_x, self.scale_x, self.bias_x, self.mean_x,
            self.var_x, z, self.filter_z, self.scale_z, self.bias_z,
            self.mean_z, self.var_z, self._stride, self._stride_z,
            self._padding, self._dilation, self._groups, self._momentum,
            self._eps, self._data_format, self._fuse_add, self._has_shortcut,
            self._use_global_stats, self._is_test, self._act)
        return out
