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
                scale_z, bias_z, mean_z, var_z, stride, stride_z, padding,
                dilation, groups, momentum, eps, conv_format, bn_format,
                fused_add, has_shortcut, use_global_stats, is_test, act):

    helper = LayerHelper('resnet_unit', **locals())
    bn_param_dtype = fluid.core.VarDesc.VarType.FP32
    bit_mask_dtype = fluid.core.VarDesc.VarType.INT32
    out = helper.create_variable_for_type_inference(x.dtype)
    bit_mask = helper.create_variable_for_type_inference(
        dtype=bit_mask_dtype, stop_gradient=True)
    # intermediate_out for x
    conv_x = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    saved_mean_x = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd_x = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    running_mean_x = mean_x
    running_var_x = var_x
    # intermediate_out for z
    conv_z = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    saved_mean_z = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd_z = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    running_mean_z = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if mean_z is None else mean_z
    running_var_z = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True) if var_z is None else var_z

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
        'stride_z': stride_z,
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
        'is_test': is_test,
        'act_type': act
    }

    outputs = {
        'Y': out,
        'BitMask': bit_mask,
        'ConvX': conv_x,
        'SavedMeanX': saved_mean_x,
        'SavedInvstdX': saved_invstd_x,
        'RunningMeanX': running_mean_x,
        'RunningVarX': running_var_x,
        'ConvZ': conv_z,
        'SavedMeanZ': saved_mean_z,
        'SavedInvstdZ': saved_invstd_z,
        'RunningMeanZ': running_mean_z,
        'RunningVarZ': running_var_z,
    }

    helper.append_op(
        type='resnet_unit', inputs=inputs, outputs=outputs, attrs=attrs)

    return out
