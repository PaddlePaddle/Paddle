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

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
import paddle
from paddle import _C_ops

__all__ = []


def fused_feedforward(x,
                      linear1_weight,
                      linear2_weight,
                      linear1_bias=None,
                      linear2_bias=None,
                      ln1_scale=None,
                      ln1_bias=None,
                      ln2_scale=None,
                      ln2_bias=None,
                      dropout1_prob=0.5,
                      dropout2_prob=0.5,
                      act_method="relu",
                      ln1_epsilon=1e-5,
                      ln2_epsilon=1e-5,
                      dropout1_implementation='upscale_in_train',
                      dropout2_implementation='upscale_in_train',
                      normalize_pre_or_post=False,
                      name=None):
    """
        the fused_feedforward operator is the same as the following pseudo code:
        residual = src;
        if normalize_pre_or_post:
            src = layer_norm(src)
        src = linear(dropout(activation(dropout(linear(src)))))
        if not normalize_pre_or_post:
            src = layer_norm(out)

        Args:
            x (Tensor): The input tensor of fused_feedforward

    """
    if in_dygraph_mode():
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_feedforward(
            x, None, None, linear1_weight, linear1_bias, linear2_weight,
            linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            'normalize_pre_or_post', normalize_pre_or_post, 'ln1_epsilon',
            ln1_epsilon, 'ln2_epsilon', ln2_epsilon, 'act_method', act_method,
            'dropout1_prob', dropout1_prob, 'dropout2_prob', dropout2_prob,
            'dropout1_implementation', dropout1_implementation,
            'dropout2_implementation', dropout2_implementation)
        return out

    helper = LayerHelper("fused_feedforward")
    dtype = x.dtype
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'fused_feedforward')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                'fused_feedforward')

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    helper.append_op(
        type='fused_feedforward',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout1_prob': dropout1_prob,
            'dropout2_prob': dropout2_prob,
            'act_method': act_method,
            'normalize_pre_or_post': normalize_pre_or_post,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
            'dropout1_implementation': dropout1_implementation,
            'dropout2_implementation': dropout2_implementation,
        })
    return out
