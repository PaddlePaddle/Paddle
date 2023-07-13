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
    bias=None,
    src_mask=None,
    sequence_lengths=None,
    rotary_tensor=None,
    beam_cache_offset=None,
    cache_kv=None,
    qkv_out_scale=None,
    out_linear_shift=None,
    out_linear_smooth=None,
    beam_size=1,
    rotary_emb_dims=0,
    mask_broadcast_num_heads=True,
    compute_bias=False,
    use_neox_rotary_style=False,
    out_linear_in_scale=-1,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
):
    if in_dynamic_mode():
        return _C_ops.masked_multihead_attention_(
            x,
            bias,
            src_mask,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            cache_kv,
            qkv_out_scale,
            out_linear_shift,
            out_linear_smooth,
            beam_size,
            rotary_emb_dims,
            mask_broadcast_num_heads,
            compute_bias,
            use_neox_rotary_style,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )

    helper = LayerHelper('masked_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    cache_kv_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    beam_cache_offset_out = helper.create_variable_for_type_inference(
        dtype="int"
    )
    inputs = {}
    inputs['x'] = x
    if bias:
        inputs['bias'] = bias
    if src_mask:
        inputs['src_mask'] = src_mask
    if sequence_lengths:
        inputs['sequence_lengths'] = sequence_lengths
    if rotary_tensor:
        inputs['rotary_tensor'] = rotary_tensor
    if beam_cache_offset:
        inputs['beam_cache_offset'] = beam_cache_offset
    if qkv_out_scale:
        inputs['qkv_out_scale'] = qkv_out_scale
    if out_linear_shift:
        inputs['out_linear_shift'] = out_linear_shift
    if out_linear_smooth:
        inputs['out_linear_smooth'] = out_linear_smooth

    outputs = {
        'out': out,
        'cache_kv_out': cache_kv_out,
        'beam_cache_offset_out': beam_cache_offset_out,
    }
    helper.append_op(
        type='masked_multihead_attention_',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'beam_size': beam_size,
            'rotary_emb_dims': rotary_emb_dims,
            'mask_broadcast_num_heads': mask_broadcast_num_heads,
            'compute_bias': compute_bias,
            'use_neox_rotary_style': use_neox_rotary_style,
            'out_linear_in_scale': out_linear_in_scale,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
        },
    )
    return (
        (out, cache_kv, beam_cache_offset_out)
        if beam_cache_offset
        else (out, cache_kv)
    )
