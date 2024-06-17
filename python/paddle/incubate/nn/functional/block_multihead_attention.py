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


def block_multihead_attention(
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    pre_key_cache=None,
    pre_value_cache=None,
    cache_k_quant_scales=None,
    cache_v_quant_scales=None,
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    qkv_out_scale=None,
    qkv_bias=None,
    out_shift=None,
    out_smooth=None,
    max_enc_len_this_time=None,
    max_dec_len_this_time=None,
    rope_emb=None,
    mask=None,
    tgt_mask=None,
    max_seq_len=-1,
    block_size=64,
    use_neox_style=False,
    use_dynamic_cachekv_quant=False,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
    out_scale=-1,
    compute_dtype="default",
):
    """
    Block Multi-head attention for text summarization.

    Args:
        qkv (Tensor): The qkv Tensor. Its shape is [token_num, 3 * num_head * head_size].
        key_cache (Tensor): The key_cache Tensor. Its shape is [max_block_num, num_head, block_size, head_size].
        value_cache (Tensor): The value_cache Tensor. Its shape is [max_block_num, num_head, block_size, head_size].
        seq_lens_encoder (Tensor): The encoder sequence lengths of the sequences in the batch. Its shape is [batchsize, 1].
        seq_lens_decoder (Tensor): The decoder sequence lengths of the sequences in the batch. Its shape is [batchsize, 1].
        seq_lens_this_time (Tensor): The real sequence lengths of the sequences in the batch. Its shape is [batchsize, 1].
        padding_offsets (Tensor): The offsets from unpadding to padding. Its shape is [token_num].
        cum_offsets (Tensor): The offsets from padding to unpadding. Its shape is [batchsize].
        cu_seqlens_q (Tensor): The cum sequence lengths of query. Its shape is [batchsize + 1, 1].
        cu_seqlens_k (Tensor): The cum sequence lengths of key. Its shape is [batchsize + 1, 1].
        block_tables (Tensor): The block tables, used to index the cache. Its shape is [batchsize, block_num_per_seq].
        pre_key_cache (Tensor): The pre caches of key. Its shape is [batchsize, num_head, pre_cache_length, head_size].
        pre_value_cache (Tensor): The pre caches of value. Its shape is [batchsize, num_head, pre_cache_length, head_size].
        cache_k_quant_scales (Tensor): The quant scales of cache key. Its shape depends on quant mode (dynamic or static). If dynamic quantization is enabled, its shape is [batchsize, num_head], otherwise its shape is [num_head].
        cache_v_quant_scales (Tensor): The quant scales of cache value. Its shape depends on quant mode (dynamic or static). If dynamic quantization is enabled, its shape is [batchsize, num_head], otherwise its shape is [num_head].
        cache_k_dequant_scales (Tensor): The dequant scales of cache key. Its shape depends on quant mode (dynamic or static). If dynamic quantization is enabled, its shape is [batchsize, num_head], otherwise its shape is [num_head].
        cache_v_dequant_scales (Tensor): The dequant scales of cache value. Its shape depends on quant mode (dynamic or static). If dynamic quantization is enabled, its shape is [batchsize, num_head], otherwise its shape is [num_head].
        qkv_out_scale (Tensor): The dequant scale of qkv, which is the input of BLHA. If the dtype of qkv is `int32`, this input will be applied. Its shape is [3 * num_head * head_size], and its dtype should be `float32`.
        qkv_bias (Tensor): The bias of qkv. Its shape is [3 * num_head * head_size].
        out_shift (Tensor): Shift bias of fmha_out, which is the 1st return value. Its shape is [num_head * head_size].
        out_smooth (Tensor): Smooth weight of fmha_out. Its shape is [num_head * head_size].
        max_enc_len_this_time (Tensor): Sentence length of the encoder this time. Its shape is [1].
        max_dec_len_this_time (Tensor): Sentence length of the decoder this time. Its shape is [1].
        rope_emb (Tensor): The RoPE embedding. Its shape is [2, batchsize, max_seq_len, 1, head_size // 2].
        mask (Tensor): The mask of qk_matmul in encoder. Its shape is [batchsize, 1, max_seq_len, max_seq_len].
        tgt_mask (Tensor): The mask of qk_matmul in decoder. Its shape is [batchsize, 1, 1, max_seq_len].
        max_seq_len (Int): The max length of the input. Default is -1.
        block_size (Int): The block_size of cache. Default is 64.
        use_neox_style (Bool): Whether neox_style RoPE is used or not. Default is False.
        use_dynamic_cachekv_quant (Bool): Whether dynamic cache kv quantization is applied or not. Default is False.
        quant_round_type (Int): The quant round type in cache kv quantization and fmha_out quantization. If 0 is set, value will be rounding to nearest ties to even. If 1 is set, value will be rounding to nearest ties away from zero.
        quant_max_bound (Float32): The max bound of float type to int type.
        quant_min_bound (Float32): The min bound of float type to int type.
        out_scale (Float32): The quant scale of fmha_out. Default is -1, which means do not apply quantization for fmha_out.
        compute_dtype (Str): A compute dtype, is used to represent the input data type. Default is "default", which means compute dtype is determined by input dtype. However, if the dtype of input is Int32, this value should be set to actual dtype of the model.
    Returns:
        Tensor|(output, qkv_out, cache_k_out, cache_v_out), which output is the output of
        block_multihead_attention layers, qkv_out is inplace with input `qkv`, cache_k_out and cache_v_out are inplace with input `cache_k` and `cache_v`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import numpy as np
            >>> import paddle
            >>> from paddle.incubate.nn.functional import block_multihead_attention
            >>> paddle.device.set_device('gpu')

            >>> def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
            ...     cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
            ...     cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
            ...     cum_offsets[1:] = cum_offsets_now
            ...     token_num = paddle.sum(seq_lens_this_time)
            ...     padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
            ...     cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
            ...     cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
            ...     for i in range(bsz):
            ...         seq_len_now = seq_lens_this_time[i]
            ...         cum_offset = cum_offsets[i]
            ...         for j in range(seq_len_now):
            ...             padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
            ...         cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
            ...         cu_seqlens_q[i + 1] = cum_seq_len
            ...         cu_seqlens_k[i + 1] = cum_seq_len
            ...     return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k

            >>> def remove_padding(seq_lens, cu_seq_lens, inputs, token_num):
            ...     bsz, num_head, seq_len, head_size = inputs.shape
            ...     output = paddle.zeros(shape=[token_num, num_head * head_size], dtype=inputs.dtype)
            ...     inputs = inputs.transpose([0, 2, 1, 3]).reshape([bsz, seq_len, -1])
            ...     for i in range(bsz):
            ...         seq_len_now = seq_lens[i]
            ...         start_idx = cu_seq_lens[i]
            ...         end_idx = cu_seq_lens[i + 1]
            ...         output[start_idx:end_idx, :] = inputs[i, :seq_len_now, :]
            ...     return output

            >>> def create_attn_mask(
            ...     mask_type,
            ...     batch_size,
            ...     seq_lens,
            ...     pre_cache_length=0,
            ... ):
            ...     max_seq_len = max(seq_lens)
            ...     mask = paddle.zeros(
            ...         [batch_size, 1, max_seq_len, max_seq_len + pre_cache_length],
            ...         dtype=mask_type,
            ...     )
            ...     mask[:, :, :, :pre_cache_length] = 1
            ...     for i in range(batch_size):
            ...         seq_len = seq_lens[i]
            ...         mask[i, 0, :seq_len, :seq_len] = (
            ...             paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type))
            ...             - 1
            ...         ) * 1e4
            ...     return mask

            >>> def naive_attention_impl(query, key, value, cache_k, cache_v, pre_cache_k, pre_cache_v, mask, scale=1.0):
            ...     batch = query.shape[0]
            ...     heads = query.shape[1]
            ...     seq_len = query.shape[2]
            ...     head_dim = query.shape[3]
            ...     kv_head = key.shape[1]
            ...     key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
            ...     key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
            ...     key = key.reshape([batch, heads, seq_len, head_dim])
            ...     if pre_cache_k is not None:
            ...         key = paddle.concat([pre_cache_k, key], axis=2)
            ...     if cache_k is not None:
            ...         key = paddle.concat([cache_k, key], axis=2)
            ...     value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
            ...     value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
            ...     value = value.reshape([batch, heads, seq_len, head_dim])
            ...     if pre_cache_v is not None:
            ...         value = paddle.concat([pre_cache_v, value], axis=2)
            ...     if cache_v is not None:
            ...         value = paddle.concat([cache_v, value], axis=2)
            ...     qk_res = paddle.matmul(query, key, transpose_y=True)
            ...     attention = qk_res * scale
            ...     if mask is not None:
            ...         attention = attention + mask
            ...     softmax_result = paddle.nn.functional.softmax(attention, -1)
            ...     result = paddle.matmul(softmax_result, value)
            ...     return result

            >>> batch_size = 2
            >>> num_head = 8
            >>> seq_len = 64
            >>> max_dec_len = 64
            >>> head_size = 64
            >>> hid_dim = num_head * head_size
            >>> block_size = 64
            >>> block_num_per_seq = (seq_len + max_dec_len + block_size - 1) // block_size
            >>> max_block_num = block_num_per_seq * batch_size
            >>> free_list = list(range(max_block_num - 1, -1, -1))
            >>> token_num = seq_len * batch_size

            >>> dtype = paddle.float16

            >>> seq_lens_encoder = paddle.to_tensor(
            ...     [
            ...         seq_len,
            ...     ]
            ...     * batch_size,
            ...     "int32",
            ... )
            >>> seq_lens_decoder = paddle.to_tensor(
            ...     [
            ...         0,
            ...     ]
            ...     * batch_size,
            ...     "int32",
            ... )
            >>> seq_lens_this_time = seq_lens_encoder
            >>> qkv_shape = (
            ...     batch_size,
            ...     num_head,
            ...     seq_len,
            ...     head_size,
            ... )
            >>> q = paddle.randn(shape=qkv_shape, dtype=dtype)
            >>> k = paddle.randn(shape=qkv_shape, dtype=dtype)
            >>> v = paddle.randn(shape=qkv_shape, dtype=dtype)
            >>> qkv = paddle.stack([q.transpose([0, 2, 1, 3]).reshape([token_num, hid_dim]),
            ...                     k.transpose([0, 2, 1, 3]).reshape([token_num, hid_dim]),
            ...                     v.transpose([0, 2, 1, 3]).reshape([token_num, hid_dim])], axis=1).reshape([token_num, -1])
            >>> scale = 1.0 / np.sqrt(head_size)
            >>> cache_shape = (max_block_num, num_head, block_size, head_size)
            >>> cache_k = paddle.zeros(cache_shape, dtype=dtype)
            >>> cache_v = paddle.zeros(cache_shape, dtype=dtype)
            >>> block_tables = paddle.zeros(shape=(batch_size, block_num_per_seq), dtype="int32")
            >>> for i in range(batch_size):
            ...     need_block_num = (seq_len + max_dec_len + block_size - 1) // block_size
            ...     for j in range(need_block_num):
            ...         block_tables[i, j] = free_list.pop()
            >>> padding_offset, cum_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset(batch_size, seq_len, seq_lens_this_time)
            >>> out  = block_multihead_attention(
            ...     qkv,
            ...     cache_k,
            ...     cache_v,
            ...     seq_lens_encoder,
            ...     seq_lens_decoder,
            ...     seq_lens_this_time,
            ...     padding_offset,
            ...     cum_offset,
            ...     cu_seqlens_q,
            ...     cu_seqlens_k,
            ...     block_tables,
            ...     None, # pre_key_cache
            ...     None, # pre_value_cache
            ...     None, # cache_k_quant_scales
            ...     None, # cache_v_quant_scales
            ...     None, # cache_k_dequant_scales
            ...     None, # cache_v_dequant_scales
            ...     None, # qkv_out_scale
            ...     None, # qkv_bias
            ...     None, # out_shift
            ...     None, # out_smooth
            ...     None, # max_enc_len_this_time
            ...     None, # max_dec_len_this_time
            ...     None, # rotary_embs
            ...     None, # attn_mask
            ...     None, # tgt_mask
            ...     seq_len,
            ...     block_size
            ... )[0]

            >>> attention_mask = create_attn_mask(
            ...     dtype,
            ...     batch_size,
            ...     [
            ...         seq_len,
            ...     ]
            ...     * batch_size,
            ... )

            >>> out_ref = naive_attention_impl(q, k, v, None, None, None, None, attention_mask, scale)
            >>> out_ref = remove_padding(seq_lens_this_time, cu_seqlens_q, out_ref, token_num)
            >>> # equals to: out_ref = out

            >>> print(out.shape) # [token_num, hid_dim]
            [128, 512]
    """

    if in_dynamic_mode():
        return _C_ops.block_multihead_attention_(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
            block_tables,
            pre_key_cache,
            pre_value_cache,
            rope_emb,
            mask,
            tgt_mask,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            qkv_out_scale,
            qkv_bias,
            out_shift,
            out_smooth,
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_seq_len,
            block_size,
            use_neox_style,
            use_dynamic_cachekv_quant,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out_scale,
            compute_dtype,
        )

    helper = LayerHelper('block_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=qkv.dtype)

    inputs = {}
    inputs['qkv'] = qkv
    inputs['key_cache'] = key_cache
    inputs['value_cache'] = value_cache
    inputs['seq_lens_encoder'] = seq_lens_encoder
    inputs['seq_lens_decoder'] = seq_lens_decoder
    inputs['seq_lens_this_time'] = seq_lens_this_time
    inputs['padding_offsets'] = padding_offsets
    inputs['cum_offsets'] = cum_offsets
    inputs['cu_seqlens_q'] = cu_seqlens_q
    inputs['cu_seqlens_k'] = cu_seqlens_k
    inputs['block_tables'] = block_tables
    if pre_key_cache is not None:
        inputs['pre_key_cache'] = pre_key_cache
    if pre_value_cache is not None:
        inputs['pre_value_cache'] = pre_value_cache
    if rope_emb is not None:
        inputs['rope_emb'] = rope_emb
    if mask is not None:
        inputs['mask'] = mask
    if tgt_mask is not None:
        inputs['tgt_mask'] = tgt_mask
    if cache_k_quant_scales is not None:
        inputs["cache_k_quant_scales"] = cache_k_quant_scales
    if cache_v_quant_scales is not None:
        inputs["cache_v_quant_scales"] = cache_v_quant_scales
    if cache_k_dequant_scales is not None:
        inputs["cache_k_dequant_scales"] = cache_k_dequant_scales
    if cache_v_dequant_scales is not None:
        inputs["cache_v_dequant_scales"] = cache_v_dequant_scales
    if qkv_out_scale is not None:
        inputs["qkv_out_scale"] = qkv_out_scale
    if qkv_bias is not None:
        inputs["qkv_bias"] = qkv_bias
    if out_shift is not None:
        inputs["out_shift"] = out_shift
    if out_smooth is not None:
        inputs["out_smooth"] = out_smooth
    if max_enc_len_this_time is not None:
        inputs["max_enc_len_this_time"] = max_enc_len_this_time
    if max_dec_len_this_time is not None:
        inputs["max_dec_len_this_time"] = max_dec_len_this_time

    outputs = {
        'fmha_out': out,
        'qkv_out': qkv,
        'key_cache_out': key_cache,
        'value_cache_out': value_cache,
    }
    helper.append_op(
        type='block_multihead_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seq_len': max_seq_len,
            'block_size': block_size,
            'use_neox_style': use_neox_style,
            'dynamic_cachekv_quant': use_dynamic_cachekv_quant,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
            'out_scale': out_scale,
            'compute_dtype': compute_dtype,
        },
    )
    return out, qkv, key_cache, value_cache


def block_multihead_attention_xpu(
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    cache_k_per_batch_maxs,
    cache_v_per_batch_maxs,
    pre_key_cache=None,
    pre_value_cache=None,
    cache_k_quant_scales=None,
    cache_v_quant_scales=None,
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    qkv_out_scale=None,
    qkv_bias=None,
    out_shift=None,
    out_smooth=None,
    max_enc_len_this_time=None,
    max_dec_len_this_time=None,
    rope_emb=None,
    mask=None,
    tgt_mask=None,
    max_seq_len=-1,
    block_size=64,
    use_neox_style=False,
    use_dynamic_cachekv_quant=False,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
    out_scale=-1,
    compute_dtype="default",
):
    if in_dynamic_mode():
        return _C_ops.block_multihead_attention_xpu(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
            block_tables,
            cache_k_per_batch_maxs,
            cache_v_per_batch_maxs,
            pre_key_cache,
            pre_value_cache,
            rope_emb,
            mask,
            tgt_mask,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            qkv_out_scale,
            qkv_bias,
            out_shift,
            out_smooth,
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_seq_len,
            block_size,
            use_neox_style,
            use_dynamic_cachekv_quant,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out_scale,
            compute_dtype,
        )

    helper = LayerHelper('block_multihead_attention_xpu', **locals())
    out = helper.create_variable_for_type_inference(dtype=qkv.dtype)

    inputs = {}
    inputs['qkv'] = qkv
    inputs['key_cache'] = key_cache
    inputs['value_cache'] = value_cache
    inputs['seq_lens_encoder'] = seq_lens_encoder
    inputs['seq_lens_decoder'] = seq_lens_decoder
    inputs['seq_lens_this_time'] = seq_lens_this_time
    inputs['padding_offsets'] = padding_offsets
    inputs['cum_offsets'] = cum_offsets
    inputs['cu_seqlens_q'] = cu_seqlens_q
    inputs['cu_seqlens_k'] = cu_seqlens_k
    inputs['block_tables'] = block_tables
    inputs['cache_k_per_batch_maxs'] = cache_k_per_batch_maxs
    inputs['cache_v_per_batch_maxs'] = cache_v_per_batch_maxs
    if pre_key_cache is not None:
        inputs['pre_key_cache'] = pre_key_cache
    if pre_value_cache is not None:
        inputs['pre_value_cache'] = pre_value_cache
    if rope_emb is not None:
        inputs['rope_emb'] = rope_emb
    if mask is not None:
        inputs['mask'] = mask
    if tgt_mask is not None:
        inputs['tgt_mask'] = tgt_mask
    if cache_k_quant_scales is not None:
        inputs["cache_k_quant_scales"] = cache_k_quant_scales
    if cache_v_quant_scales is not None:
        inputs["cache_v_quant_scales"] = cache_v_quant_scales
    if cache_k_dequant_scales is not None:
        inputs["cache_k_dequant_scales"] = cache_k_dequant_scales
    if cache_v_dequant_scales is not None:
        inputs["cache_v_dequant_scales"] = cache_v_dequant_scales
    if qkv_out_scale is not None:
        inputs["qkv_out_scale"] = qkv_out_scale
    if qkv_bias is not None:
        inputs["qkv_bias"] = qkv_bias
    if out_shift is not None:
        inputs["out_shift"] = out_shift
    if out_smooth is not None:
        inputs["out_smooth"] = out_smooth
    if max_enc_len_this_time is not None:
        inputs["max_enc_len_this_time"] = max_enc_len_this_time
    if max_dec_len_this_time is not None:
        inputs["max_dec_len_this_time"] = max_dec_len_this_time

    outputs = {
        'fmha_out': out,
        'qkv_out': qkv,
        'key_cache_out': key_cache,
        'value_cache_out': value_cache,
    }
    helper.append_op(
        type='block_multihead_attention_xpu',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seq_len': max_seq_len,
            'block_size': block_size,
            'use_neox_style': use_neox_style,
            'dynamic_cachekv_quant': use_dynamic_cachekv_quant,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
            'out_scale': out_scale,
            'compute_dtype': compute_dtype,
        },
    )
    return out, qkv, key_cache, value_cache
