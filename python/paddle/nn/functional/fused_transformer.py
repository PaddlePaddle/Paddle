#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from ...fluid.framework import in_dygraph_mode
from paddle import _C_ops

__all__ = []


def fused_multihead_attention(x,
                              qkv_weight,
                              linear_weight,
                              pre_layer_norm=False,
                              pre_ln_scale=None,
                              pre_ln_bias=None,
                              ln_scale=None,
                              ln_bias=None,
                              pre_ln_epsilon=1e-05,
                              qkv_bias=None,
                              linear_bias=None,
                              attn_mask=None,
                              dropout_rate=0.5,
                              attn_dropout_rate=0.5,
                              ln_epsilon=1e-05,
                              name=None):
    r"""
    """
    if in_dygraph_mode():
        # pre_ln_mean, pre_ln_variance, pre_ln_out, qkv_out, qkv_bias_out, transpose_out, qk_out, 
        # qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, attn_mask_out, fmha_out, 
        # linear_out, dropout_mask_out, ln_mean_out, ln_var_out, bias_dropout_residual_out, final_out
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, final_out = _C_ops.fused_attention(
            x, pre_ln_scale, pre_ln_bias, qkv_weight, qkv_bias, attn_mask,
            linear_weight, linear_bias, ln_scale, ln_bias, 'pre_layer_norm',
            pre_layer_norm, 'epsilon', pre_ln_epsilon, 'dropout_prob',
            dropout_rate, 'attn_dropout_prob', attn_dropout_rate, 'ln2epsilon',
            ln_epsilon)
        return final_out
