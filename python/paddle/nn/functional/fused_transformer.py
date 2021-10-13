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

import warnings
import paddle
from ...fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import fill_constant
from ...tensor import concat
from ...tensor.creation import zeros
from paddle.static import Variable
from ...fluid.layers import core
from ...fluid import dygraph_utils
from ...fluid.layers import unfold  # noqa: F401
from ...tensor.manipulation import squeeze
from ...tensor.manipulation import unsqueeze
from ...tensor import clip
from ...tensor import sum
from ...tensor import sqrt
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from ...fluid.framework import in_dygraph_mode, _varbase_creator

from ...fluid.framework import in_dygraph_mode
from ...fluid import core, dygraph_utils
from ...fluid import core, layers
from ...fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops

__all__ = []


def fused_multihead_attention(x,
                              qkv_weight,
                              out_linear_weight,
                              pre_layer_norm=False,
                              ln_scale=None,
                              ln_bias=None,
                              ln_2_scale=None,
                              ln_2_bias=None,
                              epsilon=1e-05,
                              qkv_bias=None,
                              out_linear_bias=None,
                              src_mask=None,
                              dropout=0.,
                              attn_dropout=0.,
                              ln2_epsilon=1e-05,
                              name=None):
    r"""
    """
    if in_dygraph_mode():
        ln_mean, ln_variance, ln_out, qkv_out, qkv_bias_out, transpose_out_2, qk_out, qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, src_mask_out, fmha_out, out_linear_out, dropout_mask_out, ln2_mean_out, ln2_var_out, bias_dropout_residual_out, final_out = _C_ops.fused_attention(
            x, ln_scale, ln_bias, qkv_weight, qkv_bias, src_mask,
            out_linear_weight, out_linear_bias, ln_2_scale, ln_2_bias,
            'pre_layer_norm', pre_layer_norm, 'epsilon', epsilon,
            'dropout_prob', dropout, 'attn_dropout_prob', attn_dropout)
        return final_out
