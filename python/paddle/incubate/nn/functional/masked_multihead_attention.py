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
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def masked_multihead_attention(
    x,
    cache_kv=None,
    src_mask=None,
    cum_offsets=None,
    sequence_lengths=None,
    rotary_tensor=None,
    beam_cache_offset=None,
    qkv_out_scale=None,
    out_linear_shift=None,
    out_linear_smooth=None,
    seq_len=1,
    rotary_emb_dims=0,
    use_neox_rotary_style=False,
    out_linear_in_scale=-1,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
):
    r"""
    Multi-head attention for text summarization.
    This is a fusion operator to compute masked multihead attention in transformer model architecture.
    This operator only supports running on GPU. The function of the transformer layer is consistent
    with the following pseudo code:

        .. code-block:: python
            import paddle

            x = paddle.rand(shape=(2, 3, 32, 128), dtype="float32")
            x = paddle.transpose(x, [0, 2, 1, 3])  # [batch\_size, sequence\_length, num\_head, dim\_head] --> [batch\_size, num\_head, sequence_length, dim\_head]
            q, k, v = paddle.split(x, 3, axis=2)
            cache_k, cache_v= paddle.split(cache_kv_out, 2, axis=0)
            k = paddle.concat([cache_k.squeeze(0), k], axis=2)
            v = paddle.concat([cache_v.squeeze(0), v], axis=2)

            product = paddle.matmul(x=q * (x.shape[3]**-0.5), y=k, transpose_y=True)
            product = product + src_mask
            product = paddle.nn.functional.softmax(product)
            out = paddle.matmul(product, v).transpose([0, 2, 1, 3])

    Args:
        x (Tensor): The input tensor could be 2-D tensor, the input data type could be float16 or float32, the shape is `[batch\_size, 3 * num\_head * dim\_head]`.
        cache_kvs (list(Tensor)|tuple(Tensor)): The cache structure tensors for the generation model, the shape is `[2, batch\_size, num\_head, max\_seq\_len, head\_dim]`.
        src_mask (Tensor): The src_mask tensor, the shape is `[batch\_size, 1, 1, sequence\_length]`.
        sequence_lengths (Tensor, optional): The sequence_lengths tensor, the shape is `[batch\_size, 1]`.
        rotary_tensor (Tensor, optional): The rotary_tensor tensor, the dtype must be float. the shape is `[batch\_size, 1, 1, sequence\_length, dim\_head]`.
        beam_cache_offset (Tensor, optional): The beam_cache_offset tensor, the shape is `[batch\_size, beam\_size, max\_seq\_len + max\_dec\_len]`.
        qkv_out_scale (Tensor, optional): The qkv_out_scale tensor, the shape is `[3, num\_head, dim\_head]`.
        out_linear_shift (Tensor, optional): The out_linear_shift tensor.
        out_linear_smooth (Tensor, optional): The out_linear_smooth tensor.
        beam_size (int, optional): The beam_size of beam search. Default 1.
        rotary_emb_dims (int, optional): The rotary_emb_dims. Default 0.
        use_neox_rotary_style (bool, optional): A flag indicating whether neox_rotary_style is needed or not. Default False.
        out_linear_in_scale (float, optional): The out_linear_in_scale.
        quant_round_type (int, optional): The quant_round_type. Default 1.
        quant_max_bound (float, optional): The quant_max_bound. Default 127.0.
        quant_min_bound (float, optional): The quant_min_bound. Default -127.0.

    Returns:
        Tensor|tuple: If "beam_cache_offset_out" is not none, return the
        tuple (output, cache_kvs_out, beam_cache_offset_out), which output is the output of
        masked_multihead_attention layers, cache_kvs_out is inplace with input `cache_kvs`.
        If "beam_cache_offset_out" is none, return the tuple (output, cache_kvs_out).

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import paddle.incubate.nn.functional as F

            # input: [batch_size, 3 * num_head * dim_head]
            x = paddle.rand(shape=(2, 3 * 32 * 128), dtype="float32")

            # src_mask: [batch_size, 1, 1, sequence_length]
            src_mask = paddle.rand(shape=(2, 1, 1, 10), dtype="float32")

            # cache_kv: [2, batch_size, num_head, max_seq_len, dim_head]
            cache_kv = paddle.rand(shape=(2, 2, 32, 64, 128), dtype="float32")

            output = F.masked_multihead_attention(
                x, src_mask=src_mask, cache_kv=cache_kv)

    """

    if in_dynamic_mode():
        return _C_ops.masked_multihead_attention_(
            x,
            cache_kv,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_linear_shift,
            out_linear_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )

    helper = LayerHelper('masked_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs['x'] = x
    inputs['cache_kv'] = cache_kv
    if src_mask:
        inputs['src_mask'] = src_mask
    if cum_offsets:
        inputs['cum_offsets'] = cum_offsets
    if sequence_lengths:
        inputs['sequence_lengths'] = sequence_lengths
    if rotary_tensor:
        inputs['rotary_tensor'] = rotary_tensor
    beam_cache_offset_flag = False
    if beam_cache_offset:
        inputs['beam_cache_offset'] = beam_cache_offset
        beam_cache_offset_flag = True
    else:
        beam_cache_offset = helper.create_variable_for_type_inference(
            dtype="int"
        )
    if qkv_out_scale:
        inputs['qkv_out_scale'] = qkv_out_scale
    if out_linear_shift:
        inputs['out_linear_shift'] = out_linear_shift
    if out_linear_smooth:
        inputs['out_linear_smooth'] = out_linear_smooth

    outputs = {
        'out': out,
        'cache_kv_out': cache_kv,
        'beam_cache_offset_out': beam_cache_offset,
    }
    helper.append_op(
        type='masked_multihead_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'seq_len': seq_len,
            'rotary_emb_dims': rotary_emb_dims,
            'use_neox_rotary_style': use_neox_rotary_style,
            'out_linear_in_scale': out_linear_in_scale,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
        },
    )
    return (
        (out, cache_kv, beam_cache_offset)
        if beam_cache_offset_flag
        else (out, cache_kv)
    )
