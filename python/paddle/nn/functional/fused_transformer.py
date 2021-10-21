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


def fused_multi_head_attention(x,
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
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces. This API only 
    support self_attention. The pseudo code is as follows:
    if pre_layer_norm:
    	out = layer_norm(x);
        out = linear(out) + qkv)bias
    else:
	out = linear(x) + bias;
    out = transpose(out, perm=[2, 0, 3, 1, 4]);
    # extract q, k and v from out.
    q = out[0:1,::]
    k = out[1:2,::]
    v = out[2:3,::]
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);      
    out = out_linear(out);
    out = layer_norm(x + dropout(linear_bias + out));

    Parameters:
        x (Tensor): The input tensor of fused_multi_head_attention. The shape is 
            `[batch\_size, sequence\_len, embed\_dim]`.
        qkv_weight (Tensor): The qkv weight tensor. The shape is `[3, num_head, dim_head, dim_embed]`.
        linear_weight (Tensor): The linear weight tensor. The shape is `[embed_dim, embed_dim]`.
        pre_layer_norm (bool, optional): whether it is pre_layer_norm or post_layer_norm architecture. 
            Default False.
        pre_ln_scale (Tensor, optional): The weight tensor of pre layernorm. Default None.
        pre_ln_bias (Tensor, optional): The bias tensor of pre layernorm. Default None.
        ln_scale (Tensor, optional): The weight tensor of layernorm. Default None.
        ln_bias (Tensor, optional): The bias tensor of layernorm. Default None.
        pre_ln_epsilon (float, optional): Small float value added to denominator of the pre layer_norm 
            to avoid dividing by zero. Default is 1e-5.
        qkv_bias (Tensor, optional): The bias of qkv computation. The shape is `[3, num_head, dim_head]`. 
            Default None.
        linear_bias (Tensor, optional): The bias of linear. The shape is `[embed_dim]`. Default None.
        attn_mask (Tensor, optional):
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention. 
            0 for no dropout. Default 0.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention. 
            0 for no dropout. Default 0.
        ln_epsilon (float, optional): Small float value added to denominator of layer_norm 
            to avoid dividing by zero. Default is 1e-5.
         
    Examples:

        .. code-block:: python
            
            # required: gpu            
            import paddle
            import paddle.nn.functional as F

            # input: [batch_size, seq_len, embed_dim]
            x = paddle.rand(shape=(2, 4, 128), dtype="float32")
            # qkv_weight: [3, num_head, dim_head, dim_embed]
            qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float32")
            # qkv_bias: [3, num_head, dim_head]
            qkv_bias = paddle.rand(shape=(3, 4, 32), dtype="float32")
            # linear_weight: [embed_dim, embed_dim]
            linear_weight = paddle.rand(shape=(128, 128), dtype="float32")
            # linear_bias: [embed_dim]
            linear_bias = paddle.rand(shape=[128], dtype="float32")
            # self attention mask: [batch_size, num_heads, seq_len, seq_len]
            attn_mask = paddle.rand(shape=(2, 4, 4, 4), dtype="float32")

            # output: [batch_size, seq_len, embed_dim]
            output = F.fused_multi_head_attention(
                x, qkv_weight, linear_weight, False,
                None, None, None, None, 1e-5, qkv_bias,
                linear_bias, attn_mask)
            # [2, 4, 128]
            print(output.shape)
    """
    if in_dygraph_mode():
        # pre_ln_mean, pre_ln_variance, pre_ln_out, qkv_out, qkv_bias_out, transpose_out, qk_out, 
        # qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, attn_mask_out, fmha_out, 
        # linear_out, dropout_mask_out, ln_mean_out, ln_var_out, bias_dropout_residual_out, final_out
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, final_out = _C_ops.fused_attention(
            x, pre_ln_scale, pre_ln_bias, qkv_weight, qkv_bias, attn_mask,
            linear_weight, linear_bias, ln_scale, ln_bias, 'pre_layer_norm',
            pre_layer_norm, 'epsilon', pre_ln_epsilon, 'dropout_rate',
            dropout_rate, 'attn_dropout_rate', attn_dropout_rate, 'ln_epsilon',
            ln_epsilon)
        return final_out
