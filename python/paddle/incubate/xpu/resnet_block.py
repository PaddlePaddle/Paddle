#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn import initializer as I
from paddle.nn import Layer, LayerList
from paddle.fluid.layers import utils
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid.param_attr import ParamAttr
from paddle import _C_ops, _legacy_C_ops

__all__ = ['resnet_basic_block', 'ResNetBasicBlock']


def resnet_basic_block(x,
                       filter1,
                       scale1,
                       bias1,
                       mean1,
                       var1,
                       filter2,
                       scale2,
                       bias2,
                       mean2,
                       var2,
                       filter3,
                       scale3,
                       bias3,
                       mean3,
                       var3,
                       stride1,
                       stride2,
                       stride3,
                       padding1,
                       padding2,
                       padding3,
                       dilation1,
                       dilation2,
                       dilation3,
                       groups,
                       momentum,
                       eps,
                       data_format,
                       has_shortcut,
                       use_global_stats=None,
                       training=False,
                       trainable_statistics=False,
                       find_conv_max=True):

    if fluid.framework._non_static_mode():
        attrs = ('stride1', stride1, 'stride2', stride2, 'stride3', stride3,
                 'padding1', padding1, 'padding2', padding2, 'padding3',
                 padding3, 'dilation1', dilation1, 'dilation2', dilation2,
                 'dilation3', dilation3, 'group', groups, 'momentum', momentum,
                 'epsilon', eps, 'data_format', data_format, 'has_shortcut',
                 has_shortcut, 'use_global_stats', use_global_stats,
                 "trainable_statistics", trainable_statistics, 'is_test',
                 not training, 'act_type', "relu", 'find_conv_input_max',
                 find_conv_max)

        out, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                _legacy_C_ops.resnet_basic_block(x, filter1, scale1, bias1, mean1, var1, filter2, scale2, bias2, mean2, var2, \
                filter3, scale3, bias3, mean3, var3, mean1, var1, mean2, var2, mean3, var3, *attrs)
        return out
    helper = LayerHelper('resnet_basic_block', **locals())
    bn_param_dtype = fluid.core.VarDesc.VarType.FP32
    max_dtype = fluid.core.VarDesc.VarType.FP32

    out = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                    stop_gradient=True)
    conv1 = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                      stop_gradient=True)
    saved_mean1 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd1 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    running_mean1 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if mean1 is None else mean1
    running_var1 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if var1 is None else var1
    conv2 = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                      stop_gradient=True)
    conv2_input = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                            stop_gradient=True)
    saved_mean2 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd2 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    running_mean2 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if mean2 is None else mean2
    running_var2 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if var2 is None else var2
    conv3 = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                      stop_gradient=True)
    saved_mean3 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd3 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    running_mean3 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if mean3 is None else mean3
    running_var3 = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if var3 is None else var3
    conv1_input_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)
    conv1_filter_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)
    conv2_input_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)
    conv2_filter_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)
    conv3_input_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)
    conv3_filter_max = helper.create_variable_for_type_inference(
        dtype=max_dtype, stop_gradient=True)

    inputs = {
        'X': x,
        'Filter1': filter1,
        'Scale1': scale1,
        'Bias1': bias1,
        'Mean1': mean1,
        'Var1': var1,
        'Filter2': filter2,
        'Scale2': scale2,
        'Bias2': bias2,
        'Mean2': mean2,
        'Var2': var2,
        'Filter3': filter3,
        'Scale3': scale3,
        'Bias3': bias3,
        'Mean3': mean3,
        'Var3': var3,
    }

    attrs = {
        'stride1': stride1,
        'stride2': stride2,
        'stride3': stride3,
        'padding1': padding1,
        'padding2': padding2,
        'padding3': padding3,
        'dilation1': dilation1,
        'dilation2': dilation2,
        'dilation3': dilation3,
        'group': groups,
        'momentum': momentum,
        'epsilon': eps,
        'data_format': data_format,
        'has_shortcut': has_shortcut,
        'use_global_stats': use_global_stats,
        "trainable_statistics": trainable_statistics,
        'is_test': not training,
        'act_type': "relu",
        'find_conv_input_max': find_conv_max
    }

    outputs = {
        'Y': out,
        'Conv1': conv1,
        'SavedMean1': saved_mean1,
        'SavedInvstd1': saved_invstd1,
        'Mean1Out': running_mean1,
        'Var1Out': running_var1,
        'Conv2': conv2,
        'SavedMean2': saved_mean2,
        'SavedInvstd2': saved_invstd2,
        'Mean2Out': running_mean2,
        'Var2Out': running_var2,
        'Conv2Input': conv2_input,
        'Conv3': conv3,
        'SavedMean3': saved_mean3,
        'SavedInvstd3': saved_invstd3,
        'Mean3Out': running_mean3,
        'Var3Out': running_var3,
        'MaxInput1': conv1_input_max,
        'MaxFilter1': conv1_filter_max,
        'MaxInput2': conv2_input_max,
        'MaxFilter2': conv2_filter_max,
        'MaxInput3': conv3_input_max,
        'MaxFilter3': conv3_filter_max,
    }
    helper.append_op(type='resnet_basic_block',
                     inputs=inputs,
                     outputs=outputs,
                     attrs=attrs)
    return out


class ResNetBasicBlock(Layer):
    """
    ResNetBasicBlock is designed for optimize the performence of the basic unit of ssd resnet block.
    The fusion op architecture like this:
            has_shortcut = True:       else:
                    X                         X
                  /                         /
                |       |                 |       |
              CONV1     |               CONV1     |
                |       |                 |       |
               BN1      |                BN1      |
                |       |                 |       |
              RELU1     |               RELU1     |
                |       |                 |       |
              CONV2   CONV3             CONV2     |
                |       |                 |       |
               BN2     BN3               BN2      |
                 \     /                   \     /
                   ADD                       ADD
                    |                         |
                   RELU                      RELU
                    |                         |
                    Y                         Y
    """

    def __init__(self,
                 num_channels1,
                 num_filter1,
                 filter1_size,
                 num_channels2,
                 num_filter2,
                 filter2_size,
                 num_channels3,
                 num_filter3,
                 filter3_size,
                 stride1=1,
                 stride2=1,
                 stride3=1,
                 act='relu',
                 momentum=0.9,
                 eps=1e-5,
                 data_format='NCHW',
                 has_shortcut=False,
                 use_global_stats=False,
                 is_test=False,
                 filter1_attr=None,
                 scale1_attr=None,
                 bias1_attr=None,
                 moving_mean1_name=None,
                 moving_var1_name=None,
                 filter2_attr=None,
                 scale2_attr=None,
                 bias2_attr=None,
                 moving_mean2_name=None,
                 moving_var2_name=None,
                 filter3_attr=None,
                 scale3_attr=None,
                 bias3_attr=None,
                 moving_mean3_name=None,
                 moving_var3_name=None,
                 padding1=0,
                 padding2=0,
                 padding3=0,
                 dilation1=1,
                 dilation2=1,
                 dilation3=1,
                 trainable_statistics=False,
                 find_conv_max=True):
        super(ResNetBasicBlock, self).__init__()
        self._stride1 = stride1
        self._stride2 = stride2
        self._kernel1_size = utils.convert_to_list(filter1_size, 2,
                                                   'filter1_size')
        self._kernel2_size = utils.convert_to_list(filter2_size, 2,
                                                   'filter2_size')
        self._dilation1 = dilation1
        self._dilation2 = dilation2
        self._padding1 = padding1
        self._padding2 = padding2
        self._groups = 1
        self._momentum = momentum
        self._eps = eps
        self._data_format = data_format
        self._act = act
        self._has_shortcut = has_shortcut
        self._use_global_stats = use_global_stats
        self._is_test = is_test
        self._trainable_statistics = trainable_statistics
        self._find_conv_max = find_conv_max

        if has_shortcut:
            self._kernel3_size = utils.convert_to_list(filter3_size, 2,
                                                       'filter3_size')
            self._padding3 = padding3
            self._stride3 = stride3
            self._dilation3 = dilation3
        else:
            self._kernel3_size = None
            self._padding3 = 1
            self._stride3 = 1
            self._dilation3 = 1

        # check format
        valid_format = {'NCHW'}
        if data_format not in valid_format:
            raise ValueError(
                "conv_format must be one of {}, but got conv_format={}".format(
                    valid_format, data_format))

        def _get_default_param_initializer(channels, kernel_size):
            filter_elem_num = np.prod(kernel_size) * channels
            std = (2.0 / filter_elem_num)**0.5
            return I.Normal(0.0, std)

        # init filter
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32
        bn1_param_shape = [1, 1, num_filter1]
        bn2_param_shape = [1, 1, num_filter2]
        filter1_shape = [num_filter1, num_channels1, filter1_size, filter1_size]
        filter2_shape = [num_filter2, num_channels2, filter2_size, filter2_size]

        self.filter_1 = self.create_parameter(
            shape=filter1_shape,
            attr=filter1_attr,
            default_initializer=_get_default_param_initializer(
                num_channels1, self._kernel1_size))
        self.scale_1 = self.create_parameter(
            shape=bn1_param_shape,
            attr=scale1_attr,
            dtype=bn_param_dtype,
            default_initializer=I.Constant(1.0))
        self.bias_1 = self.create_parameter(shape=bn1_param_shape,
                                            attr=bias1_attr,
                                            dtype=bn_param_dtype,
                                            is_bias=True)
        self.mean_1 = self.create_parameter(attr=ParamAttr(
            name=moving_mean1_name,
            initializer=I.Constant(0.0),
            trainable=False),
                                            shape=bn1_param_shape,
                                            dtype=bn_param_dtype)
        self.mean_1.stop_gradient = True
        self.var_1 = self.create_parameter(
            attr=ParamAttr(name=moving_var1_name,
                           initializer=I.Constant(1.0),
                           trainable=False),
            shape=bn1_param_shape,
            dtype=bn_param_dtype)
        self.var_1.stop_gradient = True

        self.filter_2 = self.create_parameter(
            shape=filter2_shape,
            attr=filter2_attr,
            default_initializer=_get_default_param_initializer(
                num_channels2, self._kernel2_size))
        self.scale_2 = self.create_parameter(
            shape=bn2_param_shape,
            attr=scale2_attr,
            dtype=bn_param_dtype,
            default_initializer=I.Constant(1.0))
        self.bias_2 = self.create_parameter(shape=bn2_param_shape,
                                            attr=bias2_attr,
                                            dtype=bn_param_dtype,
                                            is_bias=True)
        self.mean_2 = self.create_parameter(attr=ParamAttr(
            name=moving_mean2_name,
            initializer=I.Constant(0.0),
            trainable=False),
                                            shape=bn2_param_shape,
                                            dtype=bn_param_dtype)
        self.mean_2.stop_gradient = True
        self.var_2 = self.create_parameter(
            attr=ParamAttr(name=moving_var2_name,
                           initializer=I.Constant(1.0),
                           trainable=False),
            shape=bn2_param_shape,
            dtype=bn_param_dtype)
        self.var_2.stop_gradient = True

        if has_shortcut:
            bn3_param_shape = [1, 1, num_filter3]
            filter3_shape = [
                num_filter3, num_channels3, filter3_size, filter3_size
            ]
            self.filter_3 = self.create_parameter(
                shape=filter3_shape,
                attr=filter3_attr,
                default_initializer=_get_default_param_initializer(
                    num_channels3, self._kernel3_size))
            self.scale_3 = self.create_parameter(
                shape=bn3_param_shape,
                attr=scale3_attr,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(1.0))
            self.bias_3 = self.create_parameter(shape=bn3_param_shape,
                                                attr=bias3_attr,
                                                dtype=bn_param_dtype,
                                                is_bias=True)
            self.mean_3 = self.create_parameter(attr=ParamAttr(
                name=moving_mean3_name,
                initializer=I.Constant(0.0),
                trainable=False),
                                                shape=bn3_param_shape,
                                                dtype=bn_param_dtype)
            self.mean_3.stop_gradient = True
            self.var_3 = self.create_parameter(attr=ParamAttr(
                name=moving_var3_name,
                initializer=I.Constant(1.0),
                trainable=False),
                                               shape=bn3_param_shape,
                                               dtype=bn_param_dtype)
            self.var_3.stop_gradient = True
        else:
            self.filter_3 = None
            self.scale_3 = None
            self.bias_3 = None
            self.mean_3 = None
            self.var_3 = None

    def forward(self, x):
        out = resnet_basic_block(
            x,
            self.filter_1,
            self.scale_1,
            self.bias_1,
            self.mean_1,
            self.var_1,
            self.filter_2,
            self.scale_2,
            self.bias_2,
            self.mean_2,
            self.var_2,
            self.filter_3,
            self.scale_3,
            self.bias_3,
            self.mean_3,
            self.var_3,
            self._stride1,
            self._stride2,
            self._stride3,
            self._padding1,
            self._padding2,
            self._padding3,
            self._dilation1,
            self._dilation2,
            self._dilation3,
            self._groups,
            self._momentum,
            self._eps,
            self._data_format,
            self._has_shortcut,
            use_global_stats=self._use_global_stats,
            training=self.training,
            trainable_statistics=self._trainable_statistics,
            find_conv_max=self._find_conv_max)
        return out
