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

from paddle.fluid.layer_helper import LayerHelper


def fused_ec_moe(
    x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, act_type
):
    """
    Applies fused ec_moe kernel.
    This method requires SM_ARCH in sm75, sm80, sm86.

    Args:
        x (Tensor): the input Tensor. Its shape is [bsz, seq_len, d_model].
        gate (Tensor): the gate Tensor to choose expert. Its shape is [bsz, seq_len, e].
        bmm0_weight (Tensor): the first batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm0_bias (Tensor): the first batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        bmm1_weight (Tensor): the second batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm1_bias (Tensor): the second batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        act_type (string): the Activation Type. Currently only support `gelu`, `relu`.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import fused_ec_moe

            batch = 10
            seq_len = 128
            d_model = 1024
            d_feed_forward = d_model * 4
            num_expert = 8

            x = paddle.randn([batch, seq_len, d_model])
            gate = paddle.randn([batch, seq_len, num_expert])
            bmm0_weight = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm0_bias = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm1_weight = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm1_bias = paddle.randn([num_expert, d_model, d_feed_forward])
            out = fused_ec_moe(x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, act_type="gelu")

            print(out.shape) # [batch, seq_len, num_expert]
    """
    helper = LayerHelper('fused_moe', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='moe',
        inputs={
            'X': x,
            'Gate': gate,
            'Bmm0': bmm0_weight,
            'Bias0': bmm0_bias,
            'Bmm1': bmm1_weight,
            'Bias1': bmm1_bias,
        },
        outputs={'Out': out},
        attrs={'act_type': act_type},
    )
    return out
