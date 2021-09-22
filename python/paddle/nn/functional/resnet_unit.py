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
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.data_feeder import convert_dtype
from paddle import _C_ops
__all__ = []


def resnet_unit(x, filter_x, scale_x, bias_x, mean_x, var_x, z, filter_z,
                scale_z, bias_z, mean_z, var_z, stride, padding, dilation,
                groups, momentum, eps, conv_format, bn_format, fused_add,
                has_shortcut, use_global_stats, act):

    running_mean_x = mean_x
    running_var_x = var_x
    running_mean_z = mean_z
    running_var_z = var_z
    if fluid.framework.in_dygraph_mode():
        attrs = ('stride', stride, 'pad', padding, 'dilate', dilation, 'group',
                 groups, 'momentum', momentum, 'epsilon', eps, 'conv_format',
                 conv_format, 'bn_format', bn_format, 'fused_add', fused_add,
                 'has_shortcut', has_shortcut, 'use_global_stats',
                 use_global_stats, 'act', act)
        out_list = _C_ops.resnet_unit(
            x, filter_x, scale_x, bias_x, mean_x, var_x, z, filter_z, scale_z,
            bias_z, mean_z, var_z, running_mean_x, running_var_x,
            running_mean_z, running_var_z, *attrs)
        out = out_list[0]
    else:
        helper = LayerHelper('resnet_unit', **locals())
        # intermediate_out for x
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32
        bit_mask_dtype = fluid.core.VarDesc.VarType.INT32
        out = helper.create_variable_for_type_inference(x.dtype)
        bit_mask = helper.create_variable_for_type_inference(bit_mask_dtype)
        conv_x = helper.create_variable_for_type_inference(x.dtype)
        sum_x = helper.create_variable_for_type_inference(bn_param_dtype)
        sqsum_x = helper.create_variable_for_type_inference(bn_param_dtype)
        saved_mean_x = helper.create_variable_for_type_inference(
            dtype=bn_param_dtype, stop_gradient=True)
        saved_invstd_x = helper.create_variable_for_type_inference(
            dtype=bn_param_dtype, stop_gradient=True)
        running_mean_x = mean_x
        running_var_x = var_x
        eq_scale_x = helper.create_variable_for_type_inference(x.dtype)
        eq_bias_x = helper.create_variable_for_type_inference(x.dtype)
        conv_z = helper.create_variable_for_type_inference(
            z.dtype) if has_shortcut else None
        sum_z = helper.create_variable_for_type_inference(
            bn_param_dtype) if has_shortcut else None
        sqsum_z = helper.create_variable_for_type_inference(
            bn_param_dtype) if has_shortcut else None
        saved_mean_z = helper.create_variable_for_type_inference(
            dtype=bn_param_dtype, stop_gradient=True) if has_shortcut else None
        saved_invstd_z = helper.create_variable_for_type_inference(
            dtype=bn_param_dtype, stop_gradient=True) if has_shortcut else None
        running_mean_z = mean_z
        running_var_z = var_z
        eq_scale_z = helper.create_variable_for_type_inference(
            z.dtype) if has_shortcut else None
        eq_bias_z = helper.create_variable_for_type_inference(
            z.dtype) if has_shortcut else None

        inputs = {
            'X': x,
            'FilterX': filter_x,
            'ScaleX': scale_x,
            'BiasX': bias_x,
            'MeanX': mean_x,
            'VarX': var_x,
            'Z': z,
            'FilterZ': filter_z,
            'ScaleZ': scale_z,
            'BiasZ': bias_z,
            'MeanZ': mean_z,
            'VarZ': var_z
        }

        attrs = {
            'stride': stride,
            'pad': padding,
            'dilate': dilation,
            'group': groups,
            'momentum': momentum,
            'epsilon': eps,
            'conv_format': conv_format,
            'bn_format': bn_format,
            'fused_add': fused_add,
            'has_shortcut': has_shortcut,
            'use_global_stats': use_global_stats,
            'act': act
        }

        outputs = {
            'Y': out,
            'BitMask': bit_mask,
            'ConvX': conv_x,
            'SumX': sum_x,
            'SqSumX': sqsum_x,
            'SavedMeanX': saved_mean_x,
            'SavedInvstdX': saved_invstd_x,
            'RunningMeanX': running_mean_x,
            'RunningVarX': running_var_x,
            'EqScaleX': eq_scale_x,
            'EqBiasX': eq_bias_x,
            'ConvZ': conv_z,
            'SumZ': sum_z,
            'SqSumZ': sqsum_z,
            'SavedMeanZ': saved_mean_z,
            'SavedInvstdZ': saved_invstd_z,
            'RunningMeanZ': running_mean_z,
            'RunningVarZ': running_var_z,
            'EqScaleZ': eq_scale_z,
            'EqBiasZ': eq_bias_z
        }

        helper.append_op(
            type='resnet_unit', inputs=inputs, outputs=outputs, attrs=attrs)

    return out
