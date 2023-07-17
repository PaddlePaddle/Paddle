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
from paddle.framework import in_dynamic_mode


def fused_rotary_position_embedding(q, k, v, sin=None, cos=None):
    r"""
    Fused rotary position embedding.

    Args:
        q (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape if q must be [batch_size, seq_len, num_heads, head_dim] and head_dim must be a multiple of 2.
        k (optional|Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape if k must be [batch_size, seq_len, num_heads, head_dim] and head_dim must be a multiple of 2.
        v (optional|Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape if v must be [batch_size, seq_len, num_heads, head_dim] and head_dim must be a multiple of 2.
        sin (optional|Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape if sin must be [seq_len, head_dim] or [1, 1, seq_len, head_dim] and head_dim must be a multiple of 2.
        cos (optional|Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape if cos must be [seq_len, head_dim] or [1, 1, seq_len, head_dim] and head_dim must be a multiple of 2.

    Returns:
        out_q/out_k/out_v Tensor representing the fused rotary position embedding, has same shape and data type as `q` .


    Examples:

        ..  code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import fused_rotary_position_embedding

            q = paddle.randn([1, 1, 4, 10], dtype='float16')
            k = paddle.randn([1, 1, 4, 10], dtype='float16')
            v = paddle.randn([1, 1, 4, 10], dtype='float16')
            out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v)

            x = paddle.randn([1, 1, 1, 10], dtype='float16')
            y = paddle.randn([1, 1, 1, 10], dtype='float16')
            sin = paddle.sin(x)
            cos = paddle.cos(y)
            out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v, sin=sin, cos=cos)
    """
    if in_dynamic_mode():
        return _C_ops.fused_rotary_position_embedding(q, k, v, sin, cos)
