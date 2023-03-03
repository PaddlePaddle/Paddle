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
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def cutlass_fused_glu(x, weight, bias, act_type):
    """
    Cutlass Fused Multihead Attention.
    This method requires SM_ARCH in sm70, sm75, sm80.
    Args:
        query (Tensor): the Query Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        key (Tensor): the Key Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        value (Tensor): the Value Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        mask (Tensor): the Mask Tensor. Its shape is [batchsize, num_head, seq_len, seq_len]. And it can broadcast in each dims (which means you can set dimsize=1).
        scale (Float): the attention matrix's scale. Default is sqrt(1.0 / head_size).
        causal (Bool): whether causal masking is used or not. Default is False.
    Returns:
        Tensor: the output Tensor.
    Examples:
        .. code-block:: python
            # required: gpu
            import math
            import paddle
            from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention
            batch = 1
            num_head = 8
            seq_len = 256
            head_size = 32
            dtype = paddle.float16
            query = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            key = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            value = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            mask = paddle.randn([1, 1, 1, seq_len], dtype=dtype)
            scale = float(1.0 / math.sqrt(head_size))
            out = cutlass_fused_multi_head_attention(query, key, value, mask, scale)
            print(out.shape) # [batch, seq_len, num_head, head_size]
    """
    # requires_grad = not x.stop_gradient
    requires_grad = True
    if in_dygraph_mode():
        return _C_ops.cutlass_fused_glu(
            x, weight, bias, act_type, requires_grad
        )[0]

    helper = LayerHelper('cutlass_fused_glu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    matmul0_result = helper.create_variable_for_type_inference(dtype=x.dtype)
    matmul1_result = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cutlass_fused_glu',
        inputs={'X': x, 'Weight': weight, 'Bias': bias},
        attrs={"Act_type": act_type, "Requires_grad": requires_grad},
        outputs={
            'Out': out,
            'matmul0_result': matmul0_result,
            'matmul1_result': matmul1_result,
        },
    )
    return out
