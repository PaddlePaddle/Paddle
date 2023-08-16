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
    x,
    cache_kv=None,
    kv_input=None,
    src_mask=None,
    cum_offsets=None,
    sequence_lengths=None,
    rotary_tensor=None,
    beam_cache_offset=None,
    qkv_out_scale=None,
    out_shift=None,
    out_smooth=None,
    seq_len=1,
    rotary_emb_dims=0,
    kv_split=False,
    head_kv=1,
    use_neox_rotary_style=False,
    out_scale=-1,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
):
    r"""
    Multi-query attention for text summarization.
    This is a fusion operator to compute masked multiquery attention in transformer model architecture.
    This operator only supports running on GPU. The function of the transformer layer is consistent
    with the following pseudo code:

        .. code-block:: python

            x = paddle.transpose(x, [0, 2, 1, 3])  # [batch\_size, sequence_length, num\_head, dim\_head] --> [batch\_size, num\_head, sequence_length, dim\_head]
            q, k, v = paddle.split(x, 3, axis=2)
            cache_k, cache_v= paddle.split(cache_kv_out, 2, axis=0)
            k = paddle.concat([cache_k.squeeze(0), k], axis=2)
            v = paddle.concat([cache_v.squeeze(0), v], axis=2)

            product = paddle.matmul(x=q * (x.shape[3]**-0.5), y=k, transpose_y=True)
            product = product + src_mask
            product = paddle.nn.functional.softmax(product)
            out = paddle.matmul(product, v).transpose([0, 2, 1, 3])

    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16 or float32.if q and kv are splited, the shape
                    is `[batch\_size, num\_head\_q, dim\_head]`.if q and kv are fused, the shape is batch\_size, num\_head\_q + 2 * num_head_kv, dim\_head]
        kv_input(Tensor, optional): The kv Tensor. the shape is '[batch\_size, num\_head\_kv * 2, dim\_head]'.if q and kv are fused, the kv_input is None.
        bias (Tensor, optional): The bias tensor of qkv, the shape is `[3, num\_head, dim\_head]`.
        src_mask (Tensor): The src_mask tensor. the shape is `[batch\_size, 1, 1, sequence\_length]`.
        sequence_lengths (Tensor, optional): The sequence_lengths tensor. the shape is `[batch\_size, 1]`.
        rotary_tensor (Tensor, optional): The rotary_tensor tensor. the shape is `[batch\_size, 1]`.
        beam_cache_offset (Tensor, optaional): The rotary_tensor tensor. the shape is `[batch\_size, beam\_size, max\_seq\_len + max\_dec\_len]`.
        cache_kvs (list(Tensor)|tuple(Tensor)): The cache structure tensors for the generation model. The shape is `[2, bsz, num\_head, max\_seq\_len, head\_dim]`.
        rotary_tensor (Tensor, optional): The rotary_tensor tensor. the shape is `[batch\_size, 1, 1, sequence\_length, dim_head]`.
        qkv_out_scale (Tensor, optional): The qkv_out_scale tensor. the shape is `[3, num\_head, dim\_head]]`.
        out_shift (Tensor, optional): The out_linear_shift tensor.
        out_smooth (Tensor, optional): The out_linear_smooth tensor.
        beam_size (int, optional): The beam_size of beam search. Default 1.
        rotary_emb_dims (int, optional): The rotary_emb_dims. Default 0.
        kv_split(bool, optional): whether q and kv are splited. Default False.
        head_kv(int, optional): the kv head number. Default 1.
        mask_broadcast_num_heads (bool, optional): A flag indicating whether broadcast is needed of src_mask num_head dim or not. Default True.
        compute_bias (bool, optional): A flag indicating whether bias is computed or not. Default False.
        use_neox_rotary_style (bool, optional): A flag indicating whether neox_rotary_style is needed or not. Default False.
        out_scale (float, optional): The out_linear_in_scale.
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

            # input: [batch_size, 3, num_head, dim_head]
            x = paddle.rand(shape=(2, 32+2, 128), dtype="float32")

            # src_mask: [batch_size, 1, 1, sequence_length]
            src_mask = paddle.rand(shape=(2, 1, 1, 10), dtype="float32")

            # cache_kv: [2, batch_size, num_head, max_seq_len, dim_head]
            cache_kv = paddle.rand(shape=(2, 2, 1, 64, 128), dtype="float32")

            output = F.masked_multihead_attention(
                x, src_mask=src_mask, cache_kv=cache_kv)

    """
    if in_dynamic_mode():
        return _C_ops.masked_multiquery_attention_(
            x,
            cache_kv,
            kv_input,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,        
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            kv_split,
            head_kv,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )

    helper = LayerHelper('masked_multiquery_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {}
    inputs['x'] = x
    inputs['cache_kv'] = cache_kv
    if kv_input is not None:
        inputs['kv_input'] = kv_input
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
    if qkv_out_scale is not None:
        inputs['qkv_out_scale'] = qkv_out_scale
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
            'kv_split': kv_split,
            'head_kv': head_kv,
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
