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
