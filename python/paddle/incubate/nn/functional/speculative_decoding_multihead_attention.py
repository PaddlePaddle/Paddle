# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def speculative_decoding_multihead_attention(
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
    rope_emb=None,
    mask=None,
    qkv_bias=None,
    max_enc_len_this_time=0,
    max_dec_len_this_time=0,
    token_num_in_cache=0,
    max_seq_len=-1,
    use_neox_style=False,
    compute_dtype="default",
):
    """
    Speculative-decoding Multi-head attention for decoding multiple tokens one times.

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
        rope_emb (Tensor): The RoPE embedding. Its shape is [2, batchsize, max_seq_len, 1, head_size // 2].
        mask (Tensor): The mask of qk_matmul in encoder. Its shape is [batchsize, 1, max_seq_len, max_seq_len].
        qkv_bias (Tensor): The bias of qkv. Its shape is [3 * num_head * head_size].
        max_enc_len_this_time (Int): The max length of the encoder. Because we only support bsz=1 for speculative-sampling, so it is an integer.
        max_dec_len_this_time (Int): The max length of the decoder. Default is 0.
        token_num_in_cache (Int): The number of valid tokens in the cache. Default is -1.
        max_seq_len (Int): The max length of the input. Default is -1.
        use_neox_style (Bool): Whether neox_style RoPE is used or not. Default is False.
        compute_dtype (Str): A compute dtype, is used to represent the input data type. Default is "default", which means compute dtype is determined by input dtype. However, if the dtype of input is Int32, this value should be set to actual dtype of the model.
    Returns:
        Tensor|(output, qkv_out, cache_k_out, cache_v_out), which output is the output of
        speculative_decoding_multihead_attention layers, qkv_out is inplace with input `qkv`, cache_k_out and cache_v_out are inplace with input `cache_k` and `cache_v`.
    """

    if in_dynamic_mode():
        return _C_ops.speculative_decoding_multihead_attention_(
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
            rope_emb,
            mask,
            qkv_bias,
            max_enc_len_this_time,
            max_dec_len_this_time,
            token_num_in_cache,
            max_seq_len,
            use_neox_style,
            compute_dtype,
        )

    helper = LayerHelper('speculative_decoding_multihead_attention', **locals())
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
    inputs['max_enc_len_this_time'] = max_enc_len_this_time
    inputs['max_dec_len_this_time'] = max_dec_len_this_time

    if rope_emb is not None:
        inputs['rope_emb'] = rope_emb
    if mask is not None:
        inputs['mask'] = mask
    if qkv_bias is not None:
        inputs["qkv_bias"] = qkv_bias
    if token_num_in_cache is not None:
        inputs["token_num_in_cache"] = token_num_in_cache
    outputs = {
        'fmha_out': out,
        'qkv_out': qkv,
        'key_cache_out': key_cache,
        'value_cache_out': value_cache,
    }
    helper.append_op(
        type='speculative_decoding_multihead_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seq_len': max_seq_len,
            'use_neox_style': use_neox_style,
            'compute_dtype': compute_dtype,
        },
    )
    return out, qkv, key_cache, value_cache
