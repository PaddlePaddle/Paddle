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


import paddle
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
            >>> # head_dim = 4

            >>> paddle.seed(1204)

            >>> # q, k, v: [batch_size, seq_len, num_heads, head_dim]
            >>> q = paddle.randn([2, 2, 2, 4], dtype='float16')
            >>> k = paddle.randn([2, 2, 2, 4], dtype='float16')
            >>> v = paddle.randn([2, 2, 2, 4], dtype='float16')

            >>> # sin, cos: [1, seq_len, 1, head_dim]
            >>> x = paddle.randn([1, 2, 1, 4], dtype='float16')
            >>> y = paddle.randn([1, 2, 1, 4], dtype='float16')
            >>> sin = paddle.sin(x)
            >>> cos = paddle.cos(y)

            >>> # position_ids: [batch_size, seq_len]
            >>> position_ids = paddle.randint(high=2, shape=[2, 2], dtype='int64')

            >>> # out_q, out_k, out_v: [batch_size, seq_len, num_heads, head_dim]
            >>> out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v, sin=sin, cos=cos, position_ids=position_ids, use_neox_rotary_style=False)
            >>> print(out_q)
            Tensor(shape=[2, 2, 2, 4], dtype=float16, place=Place(gpu:0), stop_gradient=True,
                [[[[-0.44238281,  1.53222656,  0.62255859,  0.21667480],
                    [ 0.41552734, -0.20666504, -0.58496094,  0.06628418]],

                    [[-0.59228516, -0.12359619,  0.29125977, -1.81640625],
                    [ 0.38061523, -0.03918457, -1.51660156, -0.52148438]]],


                    [[[-0.52392578,  0.72021484, -0.18920898,  1.49609375],
                    [-0.00397110, -0.80419922, -0.74267578,  1.43359375]],

                    [[ 0.08770752, -0.69970703, -0.17639160, -0.28027344],
                    [ 0.00321198, -0.22937012,  0.01106262, -0.26416016]]]])
    """
    if (sin is None) or (cos is None):
        assert (
            position_ids is None
        ), "position_ids without sin/cos is not correctly supported now. if you have used this before, the result is wrong."
        assert (
            use_neox_rotary_style
        ), "rotate_half without sin/cos is not correctly supported now. if you have used this before, the result is wrong."
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
            -1,
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
            'actual_num_heads': -1,
        },
    )

    return out_q, out_k, out_v


def fused_rotary_position_embedding_qkvpacked(
    qkv,
    rotate_k=False,
    rotate_kv=False,
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
        qkv (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of qkv must be [batch_size, seq_len, num_heads/num_heads_k + 2, num_heads_k, head_dim] or [seq_len, batch_size, num_heads/num_heads_k + 2, num_heads_k, head_dim] and head_dim must be a multiple of 2, and num_heads_k must be a multiple of 2.
        rotate_k (bool, optional): Whether the partition of k should be rotated.
        rotate_kv (bool, optional): Whether the partition of k and v should be rotated.
        sin (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of sin must be [seq_len, head_dim] or [1, seq_len, 1, head_dim] and head_dim must be a multiple of 2.
        cos (Tensor, optional): The input tensor. The data type is bfloat16, float16, float32 or float64. The shape of cos must be [seq_len, head_dim] or [1, seq_len, 1, head_dim] and head_dim must be a multiple of 2.
        position_ids (Tensor, optional): The input tensor. The data type is int64. The shape of position_ids must be [batch_size, seq_len].
        use_neox_rotary_style(optional|bool): When the use_neox_rotary_style is True, every two adjacent numbers are calculated. When the use_neox_rotary_style is False, the numbers corresponding to the positions of the front half and back half segments are calculated. Default True.
        time_major(optional|bool): Whether the first dimension of the qkv input means the time steps. If time_major is True, the shape of Tensor is [seq_len, batch_size, num_heads, head_dim], otherwise [batch_size, seq_len, num_heads, head_dime]. Defaults to False. `time_steps` means the length of input sequence.
        rotary_emb_base(optional|float): the base of the rotary embedding. Default 10000.

    Returns:
        out_qkv Tensor representing the fused rotary position embedding, has same shape and data type as `qkv` .


    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_rotary_position_embedding_qkvpacked

            >>> paddle.set_device('gpu')

            >>> # batch_size = 2
            >>> # seq_len = 2
            >>> # num_heads = 2
            >>> # head_dim = 4

            >>> paddle.seed(1204)

            >>> # q, k, v: [batch_size, seq_len, num_heads, head_dim]
            >>> q = paddle.randn([2, 2, 2, 4], dtype='float16')
            >>> k = paddle.randn([2, 2, 2, 4], dtype='float16')
            >>> v = paddle.randn([2, 2, 2, 4], dtype='float16')
            >>> # qkv: [batch_size, seq_len, num_group+2, num_heads, head_dim]
            >>> qkv = paddle.stack([q, k, v], axis=2)

            >>> # sin, cos: [1, seq_len, 1, head_dim]
            >>> x = paddle.randn([1, 2, 1, 4], dtype='float16')
            >>> y = paddle.randn([1, 2, 1, 4], dtype='float16')
            >>> sin = paddle.sin(x)
            >>> cos = paddle.cos(y)

            >>> # position_ids: [batch_size, seq_len]
            >>> position_ids = paddle.randint(high=2, shape=[2, 2], dtype='int64')

            >>> # out_qkv: [batch_size, seq_len, num_group+2, num_heads, head_dim]
            >>> out_qkv = fused_rotary_position_embedding_qkvpacked(qkv, rotate_kv=True, sin=sin, cos=cos, position_ids=position_ids, use_neox_rotary_style=False)
            >>> out_q = out_qkv[:,:,0,:,:] # num_group==1
            >>> print(out_q)
            Tensor(shape=[2, 2, 2, 4], dtype=float16, place=Place(gpu:0), stop_gradient=True,
                [[[[-0.44238281,  1.53222656,  0.62255859,  0.21667480],
                    [ 0.41552734, -0.20666504, -0.58496094,  0.06628418]],

                    [[-0.59228516, -0.12359619,  0.29125977, -1.81640625],
                    [ 0.38061523, -0.03918457, -1.51660156, -0.52148438]]],


                    [[[-0.52392578,  0.72021484, -0.18920898,  1.49609375],
                    [-0.00397110, -0.80419922, -0.74267578,  1.43359375]],

                    [[ 0.08770752, -0.69970703, -0.17639160, -0.28027344],
                    [ 0.00321198, -0.22937012,  0.01106262, -0.26416016]]]])
    """
    if (sin is None) or (cos is None):
        assert (
            position_ids is None
        ), "position_ids without sin/cos is not correctly supported now. if you have used this before, the result is wrong."
        assert (
            use_neox_rotary_style
        ), "rotate_half without sin/cos is not correctly supported now. if you have used this before, the result is wrong."
    input_shape = qkv.shape
    num_heads_k = qkv.shape[-2]
    num_group = qkv.shape[-3] - 2
    num_heads = num_group * num_heads_k
    qkv = paddle.flatten(qkv, 2, 3)
    actual_num_heads = num_heads
    if rotate_kv:
        actual_num_heads += 2 * num_heads_k
    elif rotate_k:
        actual_num_heads += num_heads_k

    if in_dynamic_or_pir_mode():
        out_qkv, _, _ = _C_ops.fused_rotary_position_embedding(
            qkv,
            None,
            None,
            sin,
            cos,
            position_ids,
            use_neox_rotary_style,
            time_major,
            rotary_emb_base,
            actual_num_heads,
        )
        return out_qkv.reshape(input_shape)

    helper = LayerHelper('fused_rotary_position_embedding', **locals())
    out_qkv = helper.create_variable_for_type_inference(dtype=qkv.dtype)

    outputs = {'out_q': out_qkv}

    helper.append_op(
        type='fused_rotary_position_embedding',
        inputs={
            'q': qkv,
            'k': None,
            'v': None,
            'sin': sin,
            'cos': cos,
            'position_ids': position_ids,
        },
        outputs=outputs,
        attrs={
            'use_neox_rotary_style': use_neox_rotary_style,
            'time_major': time_major,
            'rotary_emb_base': rotary_emb_base,
            'actual_num_heads': actual_num_heads,
        },
    )

    return out_qkv.reshape(input_shape)
