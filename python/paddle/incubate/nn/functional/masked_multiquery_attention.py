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
from paddle.framework import LayerHelper, in_dynamic_mode


def masked_multiquery_attention(
    query,
    key,
    value,
    cache_kv=None,
    src_mask=None,
    cum_offsets=None,
    sequence_lengths=None,
    rotary_tensor=None,
    beam_cache_offset=None,
    out_shift=None,
    out_smooth=None,
    seq_len=1,
    rotary_emb_dims=0,
    use_neox_rotary_style=False,
    out_scale=-1,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
):
    r"""
    Multi-query attention for text summarization.
    This is a fusion operator to compute masked multiquery attention in transformer model architecture.
    This operator only supports running on GPU.

    Args:
        query (Tensor): The Query Tensor. Its shape is [batchsize, num_head, head_size].
        key (Tensor): The Key Tensor. Its shape is [batchsize, num_head, head_size].
        value (Tensor): The Value Tensor. Its shape is [batchsize, num_head, head_size].
        cache_kv (list(Tensor)|tuple(Tensor)): The cache structure tensors for the generation model. The shape is `[2, bsz, num\_head, max\_seq\_len, head\_dim]`.
        src_mask (Tensor, optional): The src_mask tensor. Its shape is `[batch_size, 1, 1, sequence_length]`.
        cum_offsets (Tensor, optional): The cum_offset tensor. Its shape is `[batch_size, 1]`.
        sequence_lengths (Tensor, optional): The sequence_lengths tensor. Its shape is `[batch_size, 1]`.
        rotary_tensor (Tensor, optional): The rotary_tensor tensor. Its shape is `[batch_size, 1, 1, sequence_length, dim_head]`.
        beam_cache_offset (Tensor, optaional): The rotary_tensor tensor. Its shape is `[batch_size, beam_size, max_seq_len + max_dec_len]`.
        out_shift (Tensor, optional): The out_linear_shift tensor, used in quant.
        out_smooth (Tensor, optional): The out_linear_smooth tensor, used in quant.
        seq_len (int, optional): The seq_len, used to get input length. Default 1
        rotary_emb_dims (int, optional): The rotary_emb_dims. Default 0.
        use_neox_rotary_style (bool, optional): A flag indicating whether neox_rotary_style is needed or not. Default False.
        out_scale (float, optional): The out_scale, used in quant.
        quant_round_type (int, optional): The quant_round_type, used in quant. Default 1.
        quant_max_bound (float, optional): The quant_max_bound, used in quant. Default 127.0.
        quant_min_bound (float, optional): The quant_min_bound, used in quant. Default -127.0.

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

            # input: [batch_size, num_head, dim_head]
            q = paddle.rand(shape=(2, 32, 128), dtype="float32")
            # group = 1
            k1 = paddle.rand(shape=(2, 1, 128), dtype="float32")
            v1 = paddle.rand(shape=(2, 1, 128), dtype="float32")
            # group = 2
            k2 = paddle.rand(shape=(2, 2, 128), dtype="float32")
            v2 = paddle.rand(shape=(2, 2, 128), dtype="float32")

            # src_mask: [batch_size, 1, 1, sequence_length]
            src_mask = paddle.rand(shape=(2, 1, 1, 10), dtype="float32")

            # cache_kv: [2, batch_size, num_head, max_seq_len, dim_head]
            cache_kv1 = paddle.rand(shape=(2, 2, 1, 64, 128), dtype="float32")
            cache_kv2 = paddle.rand(shape=(2, 2, 2, 64, 128), dtype="float32")

            output1 = F.masked_multiquery_attention(
                q, k1, v1, src_mask=src_mask, cache_kv=cache_kv1)

            output2 = F.masked_multiquery_attention(
                q, k2, v2, src_mask=src_mask, cache_kv=cache_kv2)

    """
    if in_dynamic_mode():
        return _C_ops.masked_multiquery_attention_(
            query,
            key,
            value,
            cache_kv,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )

    helper = LayerHelper('masked_multiquery_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=query.dtype)
    inputs = {}
    inputs['query'] = query
    inputs['key'] = key
    inputs['value'] = value
    inputs['cache_kv'] = cache_kv
    if src_mask is not None:
        inputs['src_mask'] = src_mask
    if sequence_lengths is not None:
        inputs['sequence_lengths'] = sequence_lengths
    if rotary_tensor is not None:
        inputs['rotary_tensor'] = rotary_tensor
    beam_cache_offset_flag = False
    if beam_cache_offset is not None:
        inputs['beam_cache_offset'] = beam_cache_offset
        beam_cache_offset_flag = True
    else:
        beam_cache_offset = helper.create_variable_for_type_inference(
            dtype="int"
        )
    if out_shift is not None:
        inputs['out_shift'] = out_shift
    if out_smooth is not None:
        inputs['out_smooth'] = out_smooth

    outputs = {
        'out': out,
        'cache_kv_out': cache_kv,
        'beam_cache_offset_out': beam_cache_offset,
    }
    helper.append_op(
        type='masked_multiquery_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'seq_len': seq_len,
            'rotary_emb_dims': rotary_emb_dims,
            'use_neox_rotary_style': use_neox_rotary_style,
            'out_scale': out_scale,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
        },
    )
    return (
        (out, cache_kv, beam_cache_offset)
        if beam_cache_offset_flag is not None
        else (out, cache_kv)
    )
