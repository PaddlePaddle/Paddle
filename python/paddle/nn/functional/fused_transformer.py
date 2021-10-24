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
from paddle.fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from ...fluid import core, dygraph_utils
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
        attn_mask (Tensor, optional):  A tensor used in multi-head attention to prevents attention to 
 	    some unwanted positions, usually the paddings or the subsequent positions. It is a tensor 
            with shape broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`. When the 
            data type is bool, the unwanted positions have `False` values and the others have `True` values. 
            When the data type is int, the unwanted positions have 0 values and the others have 1 values. 
            When the data type is float, the unwanted positions have `-INF` values and the others have 0 values. 
            It can be None when nothing wanted or needed to be prevented attention to. Default None.
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
            # qkv_weight: [3, num_head, head_dim, embed_dim]
            qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float32")
            # qkv_bias: [3, num_head, head_dim]
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
        assert len(qkv_weight.shape
                   ) == 4, "The dims of the shape of qkv_weight should be 4."
        assert qkv_weight.shape[
            0] == 3, "The shape of qkv_weight should be [3, num_head, head_dim, embed_dim]."
        assert qkv_weight.shape[3] == x.shape[
            2], "The 3rd dim of qkv_weight and 2nd dim of x should be the same, i.e., embed_dim."
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, final_out = _C_ops.fused_attention(
            x, pre_ln_scale, pre_ln_bias, qkv_weight, qkv_bias, attn_mask,
            linear_weight, linear_bias, ln_scale, ln_bias, 'pre_layer_norm',
            pre_layer_norm, 'epsilon', pre_ln_epsilon, 'dropout_rate',
            dropout_rate, 'attn_dropout_rate', attn_dropout_rate, 'ln_epsilon',
            ln_epsilon)
        return final_out
    else:
        helper = LayerHelper('fused_multi_head_attention', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'fused_multihead_attention')
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'fused_multi_head_attention')

        # set inputs
        inputs = dict()
        inputs['X'] = [x]
        if pre_ln_scale:
            inputs['LnScale'] = [pre_ln_scale]
        if pre_ln_bias:
            inputs['LnBias'] = [pre_ln_bias]
        inputs['QKVW'] = [qkv_weight]
        inputs['QKVBias'] = [qkv_bias]
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = [linear_weight]
        inputs['OutLinearBias'] = [linear_bias]
        if ln_scale:
            inputs['Ln2Scale'] = [ln_scale]
        if ln_bias:
            inputs['Ln2Bias'] = [ln_bias]

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': pre_ln_epsilon,
            'ln_epsilon': ln_epsilon,
            'dropout_rate': dropout_rate,
            'attn_dropout_rate': attn_dropout_rate
        }

        # set outputs
        pre_ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        pre_ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        pre_ln_out = helper.create_variable_for_type_inference(dtype=dtype)

        qkv_out = helper.create_variable_for_type_inference(dtype=dtype)
        qkv_bias_out = helper.create_variable_for_type_inference(dtype=dtype)

        transpose_out = helper.create_variable_for_type_inference(dtype=dtype)
        qk_out = helper.create_variable_for_type_inference(dtype=dtype)
        qktv_out = helper.create_variable_for_type_inference(dtype=dtype)
        softmax_out = helper.create_variable_for_type_inference(dtype=dtype)
        attn_dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        attn_dropout_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        attn_mask_out = helper.create_variable_for_type_inference(dtype=dtype)
        fmha_out = helper.create_variable_for_type_inference(dtype=dtype)
        out_linear_out = helper.create_variable_for_type_inference(dtype=dtype)
        dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        bias_dropout_residual_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        final_out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fused_attention',
            inputs=inputs,
            outputs={
                "LnMean": pre_ln_mean_out,
                "LnVariance": pre_ln_variance_out,
                "LnOut": pre_ln_out,
                "QKVOut": qkv_out,
                "QKVBiasOut": qkv_bias_out,
                "TransposeOut2": transpose_out,
                "QKOut": qk_out,
                "QKTVOut": qktv_out,
                "SoftmaxOut": softmax_out,
                "AttnDropoutMaskOut": attn_dropout_mask_out,
                "AttnDropoutOut": attn_dropout_out,
                "SrcMaskOut": attn_mask_out,
                "FMHAOut": fmha_out,
                "OutLinearOut": out_linear_out,
                "DropoutMaskOut": dropout_mask_out,
                "Ln2Mean": ln_mean_out,
                "Ln2Variance": ln_variance_out,
                "BiasDropoutResidualOut": bias_dropout_residual_out,
                'Y': final_out
            },
            attrs=attrs)
        return final_out
