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
from ...fluid.framework import in_dygraph_mode
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle import _C_ops

__all__ = []


def _verify_dropout_rate(dropout_rate):
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate argument should be a number")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate argument should between 0 and 1")


def fused_feedforward(x,
                      linear1_weight,
                      linear2_weight,
                      linear1_bias=None,
                      linear2_bias=None,
                      ln1_scale=None,
                      ln1_bias=None,
                      ln2_scale=None,
                      ln2_bias=None,
                      dropout1_rate=0.5,
                      dropout2_rate=0.5,
                      activation="relu",
                      ln1_epsilon=1e-5,
                      ln2_epsilon=1e-5,
                      pre_layer_norm=False,
                      name=None):
    """
    This is a fusion operator to compute feed forward layer in transformer model architecture.
    This operator only supports running on GPU. The function of the operator is consistent with
    the following pseudo code:

.. math::
    residual = src;
    if pre_layer_norm:
        src = layer_norm(src)
    src = linear(dropout(activation(dropout(linear(src)))))
    if not pre_layer_norm:
        src = layer_norm(out)

    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16, float32 or float64, the shape is`[batch\_size, sequence\_length, d_model]`.
        linear1_weight (Tensor): The weight of first linear, the data type is same as `x`, the shape is `[d\_model, dim\_feedforward]`.
        linear2_weight (Tensor): The weight of second linear, the data type is same as `x`, the shape is `[dim\_feedforward, d\_model]`.
        linear1_bias (Tensor, optional): The bias of first linear, the data type is same as `x`, the shape is `[dim_feedforward]`. Default None.
        linear2_bias (Tensor, optional): The bias of second linear, the data type is same as `x`, the shape is `[d_model]`. Default None.
        ln1_scale (Tensor, optional): the weight of first layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln1_bias (Tensor, optional): The bias of first layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        ln2_scale (Tensor, optional): The weight of second layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln2_bias (Tensor, optional): The bias of second layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        dropout1_rate (float, optional): The first dropout probability of setting units to zero. Default 0.5.
        dropout2_rate (float, optional): The second dropout probability of setting units to zero. Default 0.5.
        activation (str, optional): The activation. Default "relu".
        ln1_epsilon (float, optional): Small float of first layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        ln2_epsilon (float, optional): Small float of second layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        pre_layer_norm (bool, optional): add layer_norm in the pre-processing stage or post-processing state.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import numpy as np
            x_data = np.random.random((1, 8, 8)).astype("float32")
            linear1_weight_data = np.random.random((8, 8)).astype("float32")
            linear2_weight_data = np.random.random((8, 8)).astype("float32")
            x = paddle.to_tensor(x_data)
            linear1_weight = paddle.to_tensor(linear1_weight_data)
            linear2_weight = paddle.to_tensor(linear2_weight_data)
            out = paddle.nn.functional.fused_feedforward(x, linear1_weight, linear2_weight)
            print(out.numpy().shape)
            # (1, 8, 8)
    """
    _verify_dropout_rate(dropout1_rate)
    _verify_dropout_rate(dropout2_rate)

    if in_dygraph_mode():
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_feedforward(
            x, None, None, linear1_weight, linear1_bias, linear2_weight,
            linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            'pre_layer_norm', pre_layer_norm, 'ln1_epsilon', ln1_epsilon,
            'ln2_epsilon', ln2_epsilon, 'act_method', activation,
            'dropout1_rate', dropout1_rate, 'dropout2_rate', dropout2_rate)
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
            'dropout1_rate': dropout1_rate,
            'dropout2_rate': dropout2_rate,
            'act_method': activation,
            'pre_layer_norm': pre_layer_norm,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
        })
    return out
