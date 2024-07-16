# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.layer_helper import LayerHelper
from paddle.framework import (
    in_dynamic_mode,
)


def fused_moe(
    x,
    gate_weight,
    ffn1_weight,
    ffn1_bias,
    ffn2_weight,
    ffn2_bias,
    int8_moe_method="",
    moe_topk=2,
):
    """
    Applies fused moe kernel.
    This method requires SM_ARCH in sm75, sm80, sm86.

    Args:

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

    """
    if in_dynamic_mode():
        print("dynamic")
        final_out = _C_ops.fused_moe(
            x,
            gate_weight,
            ffn1_weight,
            ffn1_bias,
            ffn2_weight,
            ffn2_bias,
            int8_moe_method,
            moe_topk,
        )
        return final_out
    else:
        helper = LayerHelper('fused_moe', **locals())
        final_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='fused_moe',
            inputs={
                'x': x,
                'gate_weight': gate_weight,
                'ffn1_weight': ffn1_weight,
                'ffn1_bias': ffn1_bias,
                'ffn2_weight': ffn2_weight,
                'ffn2_bias': ffn2_bias,
            },
            outputs={'out': final_out},
            attrs={
                'int8_moe_method': int8_moe_method,
                'moe_topk': moe_topk,
            },
        )
        return final_out
