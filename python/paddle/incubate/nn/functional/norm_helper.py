# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def norm_helper(
    x,
    residual,
    bias,
    norm_weight,
    norm_bias,
    epsilon,
    residual_alpha,
    norm_type,
    begin_norm_axis,
):
    r"""
    Apply LayerNorm / RMSNorm kernel.

    Args:
        x (Tensor): the input Tensor..
        residual (Tensor, optional): the residual Tensor.
        bias (Tensor, optional): the bias Tensor.
        norm_weight (Tensor): the weight Tensor to affine output.
        bias (Tensor): the bias Tensor to affine output.
        epsilon (float): a small float number to avoid divide 0.
        residual_alpha (float): a factor to scale the residual input.
        norm_type (str): the normalize type, currently only accept `layernorm`, `rmsnorm`.
        begin_norm_axis (int): the begin axis to normalize.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu

    """

    if in_dygraph_mode():
        return _C_ops.norm_helper(
            x,
            residual,
            bias,
            norm_weight,
            norm_bias,
            epsilon,
            residual_alpha,
            norm_type,
            begin_norm_axis,
        )[0:2]

    helper = LayerHelper('norm_helper', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs_dict = {}
    outputs_dict['out'] = out
    if residual:
        residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        outputs_dict['residual_out'] = residual_out

    inputs = {}
    inputs['x'] = x
    if residual:
        inputs['residual'] = residual
    if bias:
        inputs['bias'] = bias

    inputs['norm_weight'] = norm_weight
    if norm_bias:
        inputs['norm_bias'] = norm_bias

    helper.append_op(
        type='norm_helper',
        inputs=inputs,
        attrs={
            "epsilon": epsilon,
            "residual_alpha": residual_alpha,
            "norm_type": norm_type,
            "begin_norm_axis": begin_norm_axis,
        },
        outputs=outputs_dict,
    )
    return out
