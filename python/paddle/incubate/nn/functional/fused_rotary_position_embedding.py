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
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode


def fused_rotary_position_embedding(
    q,
    k=None,
    v=None,
    sin=None,
    cos=None,
    position_ids=None,
    use_neox_rotary_style=True,
    time_major=False,
    rotary_emb_base=10000.0,
):
    r"""
    Fused rotary position embedding.

    Args:
        q (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of q must be [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, num_heads, head_dim] and head_dim must be a multiple of 2.
        k (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of k must be [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, num_heads, head_dim] and head_dim must be a multiple of 2.
        v (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of v must be [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, num_heads, head_dim] and head_dim must be a multiple of 2.
        sin (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of sin must be [seq_len, head_dim] or [1, seq_len, 1, head_dim] and head_dim must be a multiple of 2.
        cos (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of cos must be [seq_len, head_dim] or [1, seq_len, 1, head_dim] and head_dim must be a multiple of 2.
        position_ids (Tensor, optional): The input tensor. The data type is int64. The shape of position_ids must be [batch_size, seq_len].
        use_neox_rotary_style(optional|bool): When the use_neox_rotary_style is True, every two adjacent numbers are calculated. When the use_neox_rotary_style is False, the numbers corresponding to the positions of the front half and back half segments are calculated. Default True.
        time_major(optional|bool): Whether the first dimension of the q, k, v input means the time steps. If time_major is True, the shape of Tensor is [seq_len, batch_size, num_heads, head_dim], otherwise [batch_size, seq_len, num_heads, head_dime]. Defaults to False. `time_steps` means the length of input sequence.
        rotary_emb_base(optional|float): the base of the rotary embedding. Default 10000.

    Returns:
        out_q/out_k/out_v Tensor representing the fused rotary position embedding, has same shape and data type as `q` .


    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_rotary_position_embedding

            >>> paddle.set_device('gpu')

            >>> # batch_size = 2
            >>> # seq_len = 2
            >>> # num_heads = 2
            >>> # head_dim = 2

            >>> paddle.seed(1204)

            >>> # q, k, v: [batch_size, seq_len, num_heads, head_dim]
            >>> q = paddle.randn([2, 2, 2, 2], dtype='float16')
            >>> k = paddle.randn([2, 2, 2, 2], dtype='float16')
            >>> v = paddle.randn([2, 2, 2, 2], dtype='float16')

            >>> # sin, cos: [1, seq_len, 1, head_dim]
            >>> x = paddle.randn([1, 2, 1, 2], dtype='float16')
            >>> y = paddle.randn([1, 2, 1, 2], dtype='float16')
            >>> sin = paddle.sin(x)
            >>> cos = paddle.cos(y)

            >>> # position_ids: [batch_size, seq_len]
            >>> position_ids = paddle.randint(high=2, shape=[2, 2], dtype='int64')

            >>> # out_q, out_k, out_v: [batch_size, seq_len, num_heads, head_dim]
            >>> out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v, sin=sin, cos=cos, position_ids=position_ids, use_neox_rotary_style=False)
            >>> print(out_q)
            Tensor(shape=[2, 2, 2, 2], dtype=float16, place=Place(gpu:0), stop_gradient=True,
            [[[[-0.54931641,  0.64990234],
               [-1.08691406,  1.18261719]],
              [[ 0.57812500,  0.11749268],
               [-0.63281250,  0.15551758]]],
             [[[-0.77050781,  0.07733154],
               [-0.73730469, -0.16735840]],
              [[ 0.07116699, -0.90966797],
               [-0.03628540, -0.20202637]]]])
    """
    if (sin is None) or (cos is None):
        assert (
            position_ids is None
        ), "position_ids without sin/cos is not correctly supported now."
        assert (
            use_neox_rotary_style
        ), "rotate_half without sin/cos is not correctly supported now."

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_rotary_position_embedding(
            q,
            k,
            v,
            sin,
            cos,
            position_ids,
            use_neox_rotary_style,
            time_major,
            rotary_emb_base,
        )

    helper = LayerHelper('fused_rotary_position_embedding', **locals())
    out_q = helper.create_variable_for_type_inference(dtype=q.dtype)
    out_k = (
        helper.create_variable_for_type_inference(dtype=k.dtype) if k else None
    )
    out_v = (
        helper.create_variable_for_type_inference(dtype=v.dtype) if v else None
    )

    outputs = {'out_q': out_q}
    if out_k:
        outputs.update({'out_k': out_k})
    if out_v:
        outputs.update({'out_v': out_v})

    helper.append_op(
        type='fused_rotary_position_embedding',
        inputs={
            'q': q,
            'k': k,
            'v': v,
            'sin': sin,
            'cos': cos,
            'position_ids': position_ids,
        },
        outputs=outputs,
        attrs={
            'use_neox_rotary_style': use_neox_rotary_style,
            'time_major': time_major,
            'rotary_emb_base': rotary_emb_base,
        },
    )

    return out_q, out_k, out_v
