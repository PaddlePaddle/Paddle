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

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor


def fused_partial_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    r"""
    Applies partial rotary position embedding on the pe_head_dim portion of input.

    Args:
        x (Tensor): The input tensor. The data type is bfloat16. The shape of x must be [batch_size, seq_len, num_heads, head_dim].
        cos (Tensor): The input tensor. The data type is bfloat16. The shape of cos must be [1, seq_len, 1, pe_head_dim] and pe_head_dim must be a multiple of 2 and mustn't exceed head_dim.
        sin (Tensor): The input tensor. The data type is bfloat16. The shape of sin must be [1, seq_len, 1, pe_head_dim] and pe_head_dim must be a multiple of 2 and mustn't exceed head_dim.

    Returns:
        out: Tensor representing the fused rotary position embedding, has same shape and data type as `x` .


    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_partial_rope

            >>> paddle.set_device('gpu')
            >>> paddle.seed(2025)

            >>> # x: [batch_size, seq_len, num_heads, head_dim]
            >>> x = paddle.randn([2, 2, 2, 4], dtype='bfloat16')

            >>> # sin, cos: [1, seq_len, 1, pe_head_dim]
            >>> cos = paddle.randn([1, 2, 1, 2], dtype='bfloat16')
            >>> sin = paddle.randn([1, 2, 1, 2], dtype='bfloat16')

            >>> # out: [batch_size, seq_len, num_heads, head_dim]
            >>> out = fused_partial_rope(x, cos, sin)
            >>> print(out)
            Tensor(shape=[2, 2, 2, 4], dtype=bfloat16, place=Place(gpu:0), stop_gradient=True,
                   [[[[-0.17968750,  0.28125000, -0.34765625, -0.92187500],
                      [-0.83593750,  2.        , -0.13476562, -0.67187500]],
                     [[ 0.38281250, -0.63281250,  0.25000000, -1.03125000],
                      [-1.92187500,  2.12500000,  1.92968750, -4.21875000]]],
                    [[[-0.90625000, -1.62500000, -0.22167969, -0.68359375],
                      [-0.76562500,  0.23828125,  0.36523438,  0.53515625]],
                     [[ 0.92578125, -0.85156250, -0.75000000,  1.50000000],
                      [ 0.41992188, -1.13281250,  0.73437500, -2.18750000]]]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_partial_rope(x, cos, sin)
