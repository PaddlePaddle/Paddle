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
    rope_emb=None,
    mask=None,
    max_seq_len=-1,
    block_size=64,
    use_neox_style=False,
):
    r"""
    Block Multi-head attention for text summarization.
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
            rope_emb,
            mask,
            max_seq_len,
            block_size,
            use_neox_style,
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
    if rope_emb is not None:
        inputs['rope_emb'] = rope_emb
    if mask is not None:
        inputs['mask'] = mask

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
        },
    )
    return out, qkv, key_cache, value_cache
