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
        x (Tensor): the input Tensor. Its shape is [bsz, seq_len, d_model].
        gate (Tensor): the gate Tensor to choose expert. Its shape is [bsz, seq_len, e].
        bmm0_weight (Tensor): the first batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm0_bias (Tensor): the first batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        bmm1_weight (Tensor): the second batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm1_bias (Tensor): the second batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        int8_moe_method (string): Currently not supported.
        moe_topk: Select the top k experts for each token.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_moe

            >>> paddle.set_device('gpu')
            >>> x = paddle.randn([10, 128, 1024])
            >>> gate = paddle.randn([10, 128, 8])
            >>> bmm0_weight = paddle.randn([8, 1024, 4096])
            >>> bmm0_bias = paddle.randn([8, 1024, 4096])
            >>> bmm1_weight = paddle.randn([8, 1024, 4096])
            >>> bmm1_bias = paddle.randn([8, 1024, 4096])
            >>> moe_topk = 2
            >>> out = fused_moe(x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, int8_moe_method="", moe_topk)
            >>> print(out.shape)
            [10, 128, 1024]

    """
    if in_dynamic_mode():
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
