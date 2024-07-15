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
from paddle import _C_ops, _legacy_C_ops
from paddle.base import core
from paddle.base.data_feeder import check_dtype, check_variable_and_dtype
from paddle.base.framework import default_main_program
from paddle.base.layer_helper import LayerHelper
from paddle.framework import (
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
)

__all__ = []


def _verify_dropout_rate(dropout_rate):
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate argument should be a number")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate argument should between 0 and 1")


def fused_feedforward(
    x,
    linear1_weight,
    linear2_weight,
    linear1_bias=None,
    linear2_bias=None,
    ln1_scale=None,
    ln1_bias=None,
    ln2_scale=None,
    ln2_bias=None,
    dropout1_rate=0.5,
    dropout2_rate=0.5,
    activation="relu",
    ln1_epsilon=1e-5,
    ln2_epsilon=1e-5,
    pre_layer_norm=False,
    training=True,
    mode='upscale_in_train',
    ring_id=-1,
    add_residual=True,
    name=None,
):
    r"""
    This is a fusion operator to compute feed forward layer in transformer model architecture.
    This operator only supports running on GPU. The function of the operator is consistent with
    the following pseudo code:

    .. code-block:: text

        >>> residual = x
        >>> if pre_layer_norm:
        ...     out = layer_norm1(x)
        ...  else:
        ...     out = x
        >>> out = linear2(dropout1(activation(linear1(src))))
        >>> if add_residual:
        ...     out = residual + dropout2(out)
        ... else:
        ...     out = dropout2(out)
        >>> if not pre_layer_norm:
        ...     out = layer_norm2(out)


    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16, float32 or float64, the shape is`[batch\_size, sequence\_length, d_model]`.
        linear1_weight (Tensor): The weight of first linear, the data type is same as `x`, the shape is `[d\_model, dim\_feedforward]`.
        linear2_weight (Tensor): The weight of second linear, the data type is same as `x`, the shape is `[dim\_feedforward, d\_model]`.
        linear1_bias (Tensor, optional): The bias of first linear, the data type is same as `x`, the shape is `[dim_feedforward]`. Default None.
        linear2_bias (Tensor, optional): The bias of second linear, the data type is same as `x`, the shape is `[d_model]`. Default None.
        ln1_scale (Tensor, optional): the weight of first layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln1_bias (Tensor, optional): The bias of first layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        ln2_scale (Tensor, optional): The weight of second layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln2_bias (Tensor, optional): The bias of second layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        dropout1_rate (float, optional): The first dropout probability of setting units to zero. Default 0.5.
        dropout2_rate (float, optional): The second dropout probability of setting units to zero. Default 0.5.
        activation (str, optional): The activation. Default "relu".
        ln1_epsilon (float, optional): Small float of first layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        ln2_epsilon (float, optional): Small float of second layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        pre_layer_norm (bool, optional): add layer_norm in the pre-processing stage or post-processing state.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        ring_id (int, optional): For distributed forward in tensor model parallel, only support NCCL. Default is -1, means not using tensor parallel.
        add_residual (bool, optional): Whether add residual at the end. Default is True.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> import paddle.incubate.nn.functional as F

            >>> x = paddle.randn(shape=(1, 8, 8), dtype="float32")
            >>> linear1_weight = paddle.randn(shape=(8, 8), dtype="float32")
            >>> linear2_weight = paddle.randn(shape=(8, 8), dtype="float32")
            >>> out = F.fused_feedforward(x, linear1_weight, linear2_weight)
            >>> print(out.shape)
            [1, 8, 8]
    """
    _verify_dropout_rate(dropout1_rate)
    _verify_dropout_rate(dropout2_rate)

    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer

    if in_dynamic_or_pir_mode():
        if paddle.static.default_main_program().random_seed != 0:
            seed = paddle.static.default_main_program().random_seed

        if in_dynamic_mode():
            out, _, _, _, _, _, _, _, _, _, _ = _legacy_C_ops.fused_feedforward(
                x,
                None,
                None,
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
                ln1_scale,
                ln1_bias,
                ln2_scale,
                ln2_bias,
                'pre_layer_norm',
                pre_layer_norm,
                'ln1_epsilon',
                ln1_epsilon,
                'ln2_epsilon',
                ln2_epsilon,
                'act_method',
                activation,
                'dropout1_rate',
                dropout1_rate,
                'dropout2_rate',
                dropout2_rate,
                "is_test",
                not training,
                "dropout1_fix_seed",
                seed is not None,
                "dropout2_fix_seed",
                seed is not None,
                "dropout1_seed",
                seed if seed is not None else 0,
                "dropout2_seed",
                seed if seed is not None else 0,
                'dropout1_implementation',
                mode,
                'dropout2_implementation',
                mode,
                'add_residual',
                add_residual,
                'ring_id',
                ring_id,
            )
        else:
            dtype = x.dtype
            check_variable_and_dtype(
                x, 'x', ['float16', 'float32', 'float64'], 'fused_feedforward'
            )
            check_dtype(
                dtype,
                'dtype',
                ['float16', 'float32', 'float64'],
                'fused_feedforward',
            )

            out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_feedforward(
                x,
                None,
                None,
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
                ln1_scale,
                ln1_bias,
                ln2_scale,
                ln2_bias,
                pre_layer_norm,
                ln1_epsilon,
                ln2_epsilon,
                activation,
                dropout1_rate,
                dropout2_rate,
                mode,
                mode,
                not training,
                seed is not None,
                seed is not None,
                seed if seed is not None else 0,
                seed if seed is not None else 0,
                add_residual,
                ring_id,
            )

        return out

    helper = LayerHelper("fused_feedforward")
    dtype = x.dtype
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64'], 'fused_feedforward'
    )
    check_dtype(
        dtype, 'dtype', ['float16', 'float32', 'float64'], 'fused_feedforward'
    )

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True
    )
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True
    )
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True
    )

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='fused_feedforward',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout1_rate': dropout1_rate,
            'dropout2_rate': dropout2_rate,
            'act_method': activation,
            'pre_layer_norm': pre_layer_norm,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
            'is_test': not training,
            'dropout1_fix_seed': seed is not None,
            'dropout2_fix_seed': seed is not None,
            'dropout1_seed': seed if seed is not None else 0,
            'dropout2_seed': seed if seed is not None else 0,
            'dropout1_implementation': mode,
            'dropout2_implementation': mode,
            'add_residual': add_residual,
            'ring_id': ring_id,
        },
    )
    return out


def fused_bias_dropout_residual_layer_norm(
    x,
    residual,
    bias=None,
    ln_scale=None,
    ln_bias=None,
    dropout_rate=0.5,
    ln_epsilon=1e-5,
    training=True,
    mode='upscale_in_train',
    name=None,
):
    r"""

    The fused_bias_dropout_residual_layer_norm operator. The pseudo code is as follows:

    .. code-block:: text

        >>> y = layer_norm(residual + dropout(bias + x))

    Parameters:
        x (Tensor): The input tensor. The shape is `[*, embed\_dim]`.
        residual (Tensor): The residual tensor. The shape is same as x.
        bias (Tensor, optional): The bias of linear. The shape is `[embed_dim]`. Default None.
        ln_scale (Tensor, optional): The weight tensor of layernorm. The shape is `[embed_dim]`. Default None.
        ln_bias (Tensor, optional): The bias tensor of layernorm. The shape is `[embed_dim]`. Default None.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        ln_epsilon (float, optional): Small float value added to denominator of layer_norm
            to avoid dividing by zero. Default is 1e-5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The output Tensor, the data type and shape is same as `x`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> import paddle.incubate.nn.functional as F

            >>> # input: [batch_size, seq_len, embed_dim]
            >>> x = paddle.rand(shape=(2, 4, 128), dtype="float32")
            >>> # residual: [batch_size, seq_len, embed_dim]
            >>> residual = paddle.rand(shape=(2, 4, 128), dtype="float32")
            >>> # linear bias: [embed_dim]
            >>> bias = paddle.rand(shape=[128], dtype="float32")
            >>> # output: [batch_size, seq_len, embed_dim]
            >>> output = F.fused_bias_dropout_residual_layer_norm(
            ...     x, residual, bias)
            >>> print(output.shape)
            [2, 4, 128]

    """
    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer

    if ln_scale is not None:
        assert (
            len(ln_scale.shape) == 1
        ), "The dims of the shape of ln_scale should be 1."
        assert (
            x.shape[len(x.shape) - 1] == ln_scale.shape[0]
        ), "The dim of ln_scale must equal to the last dim of x."
    if ln_bias is not None:
        assert (
            len(ln_bias.shape) == 1
        ), "The dims of the shape of ln_bias should be 1."
        assert (
            x.shape[len(x.shape) - 1] == ln_bias.shape[0]
        ), "The dim of ln_bias must equal to the last dim of x."

    if in_dynamic_or_pir_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        final_out = _C_ops.fused_bias_dropout_residual_layer_norm(
            x,
            residual,
            bias,
            ln_scale,
            ln_bias,
            dropout_rate,
            not training,
            seed is not None,
            seed if seed is not None else 0,
            mode,
            ln_epsilon,
        )
        return final_out
    else:
        helper = LayerHelper(
            'fused_bias_dropout_residual_layer_norm', **locals()
        )
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64'],
            'fused_bias_dropout_residual_layer_norm',
        )
        check_dtype(
            dtype,
            'dtype',
            ['float16', 'float32', 'float64'],
            'fused_bias_dropout_residual_layer_norm',
        )
        # set inputs
        inputs = {}
        inputs['X'] = [x]
        inputs['Residual'] = [residual]
        if bias is not None:
            inputs['Bias'] = [bias]
        if ln_scale:
            inputs['LnScale'] = [ln_scale]
        if ln_bias:
            inputs['LnBias'] = [ln_bias]
        if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
            seed = helper.main_program.random_seed
        # set attrs
        attrs = {
            'ln_epsilon': ln_epsilon,
            'dropout_rate': dropout_rate,
            'is_test': not training,
            'dropout_fix_seed': seed is not None,
            'dropout_seed': seed if seed is not None else 0,
            'dropout_implementation': mode,
        }
        # set outputs
        dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
        )
        ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        bias_dropout_residual_out = helper.create_variable_for_type_inference(
            dtype=dtype
        )
        final_out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fused_bias_dropout_residual_layer_norm',
            inputs=inputs,
            outputs={
                "BiasDropoutResidualOut": bias_dropout_residual_out,
                "DropoutMaskOut": dropout_mask_out,
                "LnMean": ln_mean_out,
                "LnVariance": ln_variance_out,
                'Y': final_out,
            },
            attrs=attrs,
        )
        return final_out


def fused_multi_head_attention(
    x,
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
    cache_kv=None,
    attn_mask=None,
    dropout_rate=0.5,
    attn_dropout_rate=0.5,
    ln_epsilon=1e-05,
    training=True,
    mode='upscale_in_train',
    ring_id=-1,
    add_residual=True,
    num_heads=-1,
    transpose_qkv_wb=False,
    name=None,
):
    r"""
    Attention maps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces. This API only
    support self_attention. The pseudo code is as follows:

    .. code-block:: text

        >>> residual = x
        >>> if pre_layer_norm:
        ...     out = layer_norm(x)
        ... else:
        ...     out = x
        >>> # compute q, k, v
        >>> out = matmul(out, qkv_weight) + qkv_bias
        >>> out = transpose(out, perm=[2, 0, 3, 1, 4])
        >>> # extract q, k and v from out
        >>> q = out[0:1,::] * (head_dim ** -0.5)
        >>> k = out[1:2,::]
        >>> v = out[2:3,::]
        >>> out = matmul(q, k, transpose_y=True)
        >>> out = out + attn_mask
        >>> out = softmax(out)
        >>> out = dropout(out)
        >>> out = matmul(out, v)
        >>> # combine heads
        >>> out = transpose(out, perm=[0, 2, 1, 3])
        >>> # project to output
        >>> out = linear(out)
        >>> if add_residual:
        ...     out = residual + dropout(out)
        ... else:
        ...     out = dropout(out)
        >>> if not pre_layer_norm:
        ...     out = layer_norm(out)


    Parameters:
        x (Tensor): The input tensor of fused_multi_head_attention. The shape is
            `[batch\_size, sequence\_len, embed\_dim]`.
        qkv_weight (Tensor): The qkv weight tensor. If `transpose_qkv_wb` is False, the shape is `[3, num_head, dim_head, dim_embed]`. Otherwise, the shape is `[dim_embed, 3 * dim_embed]`.
        linear_weight (Tensor): The linear weight tensor. The shape is `[embed_dim, embed_dim]`.
        pre_layer_norm (bool, optional): whether it is pre_layer_norm (True) or post_layer_norm architecture
                                        (False). Default False.
        pre_ln_scale (Tensor, optional): The weight tensor of pre layernorm. Default None.
        pre_ln_bias (Tensor, optional): The bias tensor of pre layernorm. Default None.
        ln_scale (Tensor, optional): The weight tensor of layernorm. Default None.
        ln_bias (Tensor, optional): The bias tensor of layernorm. Default None.
        pre_ln_epsilon (float, optional): Small float value added to denominator of the pre layer_norm
            to avoid dividing by zero. Default is 1e-5.
        qkv_bias (Tensor, optional): The bias of qkv computation. If `transpose_qkv_wb` is False, the shape is `[3, num_head, dim_head]`. Otherwise, the shape is `[3 * dim_embed]`.
            Default None.
        linear_bias (Tensor, optional): The bias of linear. The shape is `[embed_dim]`. Default None.
        cache_kv (Tensor, optional): For generation model, cache structure. The shape is `[2, bsz, num_head, seq_len, head_dim]`. Default None.
        attn_mask (Tensor, optional):  A tensor used in multi-head attention to prevents attention to
            some unwanted positions, usually the paddings or the subsequent positions. It is a tensor
            with shape broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`. When the
            data type is bool, the unwanted positions have `False` values and the others have `True` values.
            When the data type is int, the unwanted positions have 0 values and the others have 1 values.
            When the data type is float, the unwanted positions have `-INF` values and the others have 0 values.
            It can be None when nothing wanted or needed to be prevented attention to. Default None.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        ln_epsilon (float, optional): Small float value added to denominator of layer_norm
            to avoid dividing by zero. Default is 1e-5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        ring_id (int, optional): For distributed forward in mp, only support NCCL and forward. Default is -1, means not using mp
        add_residual (bool, optional): Whether add residual at the end. Default is True.
        num_heads (int, optional): If enable transpose_qkv_wb, should provide the num_heads. Default is -1, means not transpose qkv wb.
        transpose_qkv_wb (bool, optional): Whether transpose the qkv_weight and qkv_bias in the op. Only support GPU for now. Default is false, means not transpose qkv wb.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> import paddle.incubate.nn.functional as F

            >>> # input: [batch_size, seq_len, embed_dim]
            >>> x = paddle.rand(shape=(2, 4, 128), dtype="float32")
            >>> # qkv_weight: [3, num_head, head_dim, embed_dim]
            >>> qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float32")
            >>> # qkv_bias: [3, num_head, head_dim]
            >>> qkv_bias = paddle.rand(shape=(3, 4, 32), dtype="float32")
            >>> # linear_weight: [embed_dim, embed_dim]
            >>> linear_weight = paddle.rand(shape=(128, 128), dtype="float32")
            >>> # linear_bias: [embed_dim]
            >>> linear_bias = paddle.rand(shape=[128], dtype="float32")
            >>> # self attention mask: [batch_size, num_heads, seq_len, seq_len]
            >>> attn_mask = paddle.rand(shape=(2, 4, 4, 4), dtype="float32")

            >>> # output: [batch_size, seq_len, embed_dim]
            >>> output = F.fused_multi_head_attention(
            ...     x, qkv_weight, linear_weight, False,
            ...     None, None, None, None, 1e-5, qkv_bias,
            ...     linear_bias, None, attn_mask)
            >>> print(output.shape)
            [2, 4, 128]
    """

    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer

    if x.ndim != 3:
        raise ValueError(
            f"The rank of the x should be 3, but received {x.ndim}."
        )

    if in_dynamic_or_pir_mode():
        if paddle.static.default_main_program().random_seed != 0:
            seed = paddle.static.default_main_program().random_seed
        # pre_ln_mean, pre_ln_variance, pre_ln_out, qkv_out, qkv_bias_out, transpose_out, qk_out,
        # qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, attn_mask_out, fmha_out,
        # linear_out, dropout_mask_out, ln_mean_out, ln_var_out, bias_dropout_residual_out, final_out
        if not transpose_qkv_wb:
            assert (
                len(qkv_weight.shape) == 4
            ), "The dims of the shape of qkv_weight should be 4."
            assert (
                qkv_weight.shape[0] == 3
            ), "The shape of qkv_weight should be [3, num_head, head_dim, embed_dim]."
            assert (
                qkv_weight.shape[3] == x.shape[2]
            ), "The 3rd dim of qkv_weight and 2nd dim of x should be the same, i.e., embed_dim."
            if ring_id == -1:
                # under mp, the num head will be split, this equation will not hold
                assert (
                    qkv_weight.shape[1] * qkv_weight.shape[2]
                    == qkv_weight.shape[3]
                ), "embed_dim must be divisible by num_heads."
        else:
            assert (
                num_heads > 0
            ), "When enable transpose_qkv_wb, the num_heads should be provided and greater than 0."
            assert len(qkv_weight.shape) == 2, (
                "When enable transpose_qkv_wb, the dims of the shape of qkv_weight "
                "should be 2 when enable transpose_qkv_wb."
            )
            if ring_id == -1:
                # under mp, the num head will be split, this equation will not hold
                assert qkv_weight.shape[1] == 3 * qkv_weight.shape[0], (
                    "When enable transpose_qkv_wb, the shape of qkv_weight should be "
                    "[embed_dim, 3 * embed_dim] when enable transpose_qkv_wb."
                )
            assert qkv_weight.shape[0] == x.shape[2], (
                "When enable transpose_qkv_wb, the 1st dim of qkv_weight and 2nd dim of x "
                "should be the same, i.e., embed_dim."
            )
            if qkv_bias is not None:
                assert (
                    len(qkv_bias.shape) == 1
                ), "When enable transpose_qkv_wb, the dims of the shape of qkv_bias should be 1."
                assert qkv_bias.shape[0] == qkv_weight.shape[1], (
                    "When enable transpose_qkv_wb, the 1st dim of qkv_bias and 2nd dim of "
                    "qkv_weight should be the same, i.e., embed_dim."
                )
        if in_dynamic_mode():
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                cache_kv_out,
                final_out,
            ) = _legacy_C_ops.fused_attention(
                x,
                pre_ln_scale,
                pre_ln_bias,
                qkv_weight,
                qkv_bias,
                cache_kv,
                attn_mask,
                linear_weight,
                linear_bias,
                ln_scale,
                ln_bias,
                'num_heads',
                num_heads,
                'transpose_qkv_wb',
                transpose_qkv_wb,
                'pre_layer_norm',
                pre_layer_norm,
                'epsilon',
                pre_ln_epsilon,
                'dropout_rate',
                dropout_rate,
                'attn_dropout_rate',
                attn_dropout_rate,
                'ln_epsilon',
                ln_epsilon,
                'is_test',
                not training,
                'attn_dropout_fix_seed',
                seed is not None,
                'dropout_fix_seed',
                seed is not None,
                'attn_dropout_seed',
                seed if seed is not None else 0,
                'dropout_seed',
                seed if seed is not None else 0,
                'attn_dropout_implementation',
                mode,
                'dropout_implementation',
                mode,
                'add_residual',
                add_residual,
                'ring_id',
                ring_id,
            )
        else:
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                cache_kv_out,
                final_out,
            ) = _C_ops.fused_attention(
                x,
                pre_ln_scale,
                pre_ln_bias,
                qkv_weight,
                qkv_bias,
                cache_kv,
                attn_mask,
                linear_weight,
                linear_bias,
                ln_scale,
                ln_bias,
                num_heads,
                transpose_qkv_wb,
                pre_layer_norm,
                pre_ln_epsilon,
                attn_dropout_rate,
                not training,
                seed is not None,
                seed if seed is not None else 0,
                mode,
                dropout_rate,
                seed is not None,
                seed if seed is not None else 0,
                mode,
                ln_epsilon,
                add_residual,
                ring_id,
            )

        if cache_kv is not None:
            return final_out, cache_kv_out
        return final_out
    else:
        helper = LayerHelper('fused_multi_head_attention', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64'],
            'fused_multihead_attention',
        )
        check_dtype(
            dtype,
            'dtype',
            ['float16', 'float32', 'float64'],
            'fused_multi_head_attention',
        )

        # set inputs
        inputs = {}
        inputs['X'] = [x]
        if pre_ln_scale:
            inputs['LnScale'] = [pre_ln_scale]
        if pre_ln_bias:
            inputs['LnBias'] = [pre_ln_bias]
        inputs['QKVW'] = [qkv_weight]
        if qkv_bias is not None:
            inputs['QKVBias'] = [qkv_bias]
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = [linear_weight]
        if linear_bias is not None:
            inputs['OutLinearBias'] = [linear_bias]
        if ln_scale:
            inputs['Ln2Scale'] = [ln_scale]
        if ln_bias:
            inputs['Ln2Bias'] = [ln_bias]
        if cache_kv:
            inputs['CacheKV'] = [cache_kv]

        if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
            seed = helper.main_program.random_seed

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': pre_ln_epsilon,
            'ln_epsilon': ln_epsilon,
            'dropout_rate': dropout_rate,
            'attn_dropout_rate': attn_dropout_rate,
            'is_test': not training,
            'attn_dropout_fix_seed': seed is not None,
            'dropout_fix_seed': seed is not None,
            'attn_dropout_seed': seed if seed is not None else 0,
            'dropout_seed': seed if seed is not None else 0,
            'attn_dropout_implementation': mode,
            'dropout_implementation': mode,
            'add_residual': add_residual,
            'ring_id': ring_id,
            'num_heads': num_heads,
            'transpose_qkv_wb': transpose_qkv_wb,
        }

        # set outputs
        pre_ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        pre_ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        pre_ln_out = helper.create_variable_for_type_inference(dtype=dtype)

        qkv_out = helper.create_variable_for_type_inference(dtype=dtype)
        qkv_bias_out = helper.create_variable_for_type_inference(dtype=dtype)

        transpose_out = helper.create_variable_for_type_inference(dtype=dtype)
        qk_out = helper.create_variable_for_type_inference(dtype=dtype)
        qktv_out = helper.create_variable_for_type_inference(dtype=dtype)
        softmax_out = helper.create_variable_for_type_inference(dtype=dtype)
        attn_dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
        )
        attn_dropout_out = helper.create_variable_for_type_inference(
            dtype=dtype
        )
        attn_mask_out = helper.create_variable_for_type_inference(dtype=dtype)
        fmha_out = helper.create_variable_for_type_inference(dtype=dtype)
        out_linear_out = helper.create_variable_for_type_inference(dtype=dtype)
        dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
        )
        ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True
        )
        bias_dropout_residual_out = helper.create_variable_for_type_inference(
            dtype=dtype
        )
        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        cache_kv_out = helper.create_variable_for_type_inference(dtype=dtype)

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
                'Y': final_out,
                'CacheKVOut': cache_kv_out,
            },
            attrs=attrs,
        )

        return (final_out, cache_kv_out) if cache_kv else final_out


def fused_multi_transformer(
    x,
    ln_scales,
    ln_biases,
    qkv_weights,
    qkv_biases,
    linear_weights,
    linear_biases,
    ffn_ln_scales,
    ffn_ln_biases,
    ffn1_weights,
    ffn1_biases,
    ffn2_weights,
    ffn2_biases,
    pre_layer_norm=True,
    epsilon=1e-05,
    residual_alpha=1.0,
    cache_kvs=None,
    beam_offset=None,
    pre_caches=None,
    seq_lens=None,
    rotary_embs=None,
    time_step=None,
    attn_mask=None,
    dropout_rate=0.0,
    rotary_emb_dims=0,
    activation="gelu",
    training=False,
    mode='upscale_in_train',
    trans_qkvw=True,
    ring_id=-1,
    norm_type="layernorm",
    use_neox_rotary_style=False,
    gqa_group_size=-1,
    name=None,
):
    r"""
    This is a fusion operator to compute multi transformer layers in transformer model architecture.
    This operator only supports running on GPU. The function of the transformer layer is consistent
    with the following pseudo code:

    .. code-block:: text

        >>> if pre_layer_norm:
        ...     out = layer_norm(x)
        ...     out = qkv_linear(out) + qkv_bias
        ... else:
        ...     out = qkv_linear(x) + qkv_bias
        >>> out = transpose(out, perm=[2, 0, 3, 1, 4])
        >>> # extract q, k and v from out.
        >>> q = out[0:1, ::]
        >>> k = out[1:2, ::]
        >>> v = out[2:3, ::]
        >>> out = q * k^t
        >>> out = attn_mask + out
        >>> out = softmax(out)
        >>> out = dropout(out)
        >>> out = out * v
        >>> out = transpose(out, perm=[0, 2, 1, 3])
        >>> out = linear(out)
        >>> if pre_layer_norm:
        ...     out = x + dropout(out + bias)
        ... else:
        ...     out = layer_norm(x + dropout(out + bias))

        >>> residual = out;
        >>> if pre_layer_norm:
        ...     out = ffn_layer_norm(out)
        >>> out = ffn1_linear(out)
        >>> out = dropout(activation(out + ffn1_bias))
        >>> out = ffn2_linear(out)
        >>> out = residual + dropout(out + ffn2_bias)
        >>> if not pre_layer_norm:
        ...     out = ffn_layer_norm(out)

    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16,
            the shape is `[batch\_size, sequence\_length, d\_model]`.
        ln_scales (list(Tensor)|tuple(Tensor)): The weight tensors of attention layer_norm,
            the shape is `[d\_model]`.
        ln_biases (list(Tensor)|tuple(Tensor)): The bias tensors of attention layer_norm.
            the shape is `[d\_model]`.
        qkv_weights (list(Tensor)|tuple(Tensor)): The weight tensors of attention qkv computation.
            The shape is `[3, num\_head, dim\_head, d\_model]`.
        qkv_biases (list(Tensor)|tuple(Tensor)|None): The bias tensors of attention qkv computation.
            The shape is `[3, num\_head, dim\_head]`.
        linear_weights (list(Tensor)|tuple(Tensor)): The weight tensors of attention linear.
            The shape is `[num\_head * dim\_head, d\_model]`.
        linear_biases (list(Tensor)|tuple(Tensor)|None): The bias tensors of attention linear.
            The shape is `[d\_model]`.
        ffn_ln_scales (list(Tensor)|tuple(Tensor)): The weight tensors of feedforward layer_norm,
            the shape is `[d\_model]`
        ffn_ln_biases (list(Tensor)|tuple(Tensor)): The bias tensors of feedforward layer_norm,
            the shape is `[d\_model]`
        ffn1_weights (list(Tensor)|tuple(Tensor)): The weight tensors of feedforward first linear,
            the shape is `[d\_model, dim\_feedforward]`.
        ffn1_biases (list(Tensor)|tuple(Tensor)|None): The bias tensors of feedforward first linear,
            the shape is `[dim\_feedforward]`.
        ffn2_weights (list(Tensor)|tuple(Tensor)): The weight tensors of feedforward second linear,
            the shape is `[dim\_feedforward, d\_model]`.
        ffn2_biases (list(Tensor)|tuple(Tensor)|None): The bias tensors of feedforward second linear,
            the shape is `[d_model]`.
        pre_layer_norm (bool, optional): whether it is pre_layer_norm(True) or post_layer_norm(False).
            Default True.
        epsilon (float, optional): Small float value added to denominator of the layer_norm
            to avoid dividing by zero. Default is 1e-5.
        cache_kvs (list(Tensor)|tuple(Tensor), optional):
            The cache structure tensors for the generation model.
            The shape is `[2, bsz, num\_head, max\_seq\_len, head\_dim]`. Default None.
        pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches for the generation model.
            The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`. Default None.
        seq_lens (Tensor optional): The sequence lengths of this batch. The shape is `[bsz]`. Default None.
        rotary_embs (Tensor optional): The RoPE embs for rotary computation.
            The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
        time_step (Tensor, optional): The time step tensor for the generation model.
            Which used in decode stage, to represent the time step, that is, the real seq_len of CacheKV.
            The shape is `[1]`, must be in CPUPlace. Default None.
        attn_mask (Tensor, optional):  A tensor used in multi-head attention to prevents attention to
            some unwanted positions, usually the paddings or the subsequent positions. It is a tensor
            with shape `[batch_size, 1, sequence_length, sequence_length]`. Default None.
        dropout_rate (float, optional): The dropout probability of setting units to zero. Default 0.0.
        rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation,
            and it is 0 when rotary_embs is None,
            1 when rotary_embs is not None and pos_extra_ids is None,
            2 when rotary_embs and pos_extra_ids are both not None. Default 0.
        activation (str, optional): The activation. Default "gelu".
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default False.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        trans_qkvw (bool, optional): Whether to transpose for weights of qkv.
            If true, the shape eights of qkv should be [3, num_head, dim_head, dim_embed].
            Otherwise the shape of weights of qkv should be [dim_embed, 3, num_head, dim_head]. Default True.
        ring_id (int, optional): For distributed forward in tensor model parallel, only support NCCL.
            Default is -1, means not using mp.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor|tuple: If `cache_kvs` is None, return a tensor that has
        the same shape and data type with `x`, representing the output
        of Transformer layers. If `cache_kvs` is not None, return the
        tuple (output, cache_kvs), which output is the output of
        Transformer layers, cache_kvs is inplace with input `cache_kvs`.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('Depends on Flash Attention 2.')
            >>> import re
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> import paddle.incubate.nn.functional as F

            >>> # input: [batch_size, seq_len, embed_dim]
            >>> x = paddle.rand(shape=(2, 4, 128), dtype="float16")

            >>> # ln_scale: [embed_dim], ln_bias: [embed_dim]
            >>> ln_scale = paddle.rand(shape=(128,), dtype="float32")
            >>> ln_bias = paddle.rand(shape=(128,), dtype="float32")

            >>> # qkv_weight: [3, num_head, head_dim, embed_dim], qkv_bias: [3, num_head, head_dim]
            >>> qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float16")
            >>> qkv_bias = paddle.rand(shape=(3, 4, 32), dtype="float16")

            >>> # linear_weight: [embed_dim, embed_dim], linear_bias: [embed_dim]
            >>> linear_weight = paddle.rand(shape=(128, 128), dtype="float16")
            >>> linear_bias = paddle.rand(shape=(128,), dtype="float16")

            >>> # ffn_ln_scale: [embed_dim], ffn_ln_bias: [embed_dim]
            >>> ffn_ln_scale = paddle.rand(shape=(128,), dtype="float32")
            >>> ffn_ln_bias = paddle.rand(shape=(128,), dtype="float32")

            >>> # ffn1_weight: [embed_dim, 4*embed_dim], ffn1_bias: [4*embed_dim]
            >>> ffn1_weight = paddle.rand(shape=(128, 4*128), dtype="float16")
            >>> ffn1_bias = paddle.rand(shape=(4*128,), dtype="float16")

            >>> # ffn2_weight: [4*embed_dim, embed_dim], ffn2_bias: [embed_dim]
            >>> ffn2_weight = paddle.rand(shape=(4*128, 128), dtype="float16")
            >>> ffn2_bias = paddle.rand(shape=(128,), dtype="float16")

            >>> # self attention mask: [batch_size, 1, seq_len, seq_len]
            >>> attn_mask = paddle.rand(shape=(2, 1, 4, 4), dtype="float32")

            >>> # output: [batch_size, seq_len, embed_dim]
            >>> output = F.fused_multi_transformer(
            ...     x, [ln_scale], [ln_bias], [qkv_weight], [qkv_bias],
            ...     [linear_weight], [linear_bias], [ffn_ln_scale], [ffn_ln_bias],
            ...     [ffn1_weight], [ffn1_bias], [ffn2_weight], [ffn2_bias],
            ...     attn_mask=attn_mask)
            >>> print(output.shape)
            [2, 4, 128]
    """
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer

    if in_dynamic_or_pir_mode():
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer(
            x,
            ln_scales,
            ln_biases,
            qkv_weights,
            qkv_biases,
            cache_kvs,
            pre_caches,
            rotary_embs,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            linear_weights,
            linear_biases,
            ffn_ln_scales,
            ffn_ln_biases,
            ffn1_weights,
            ffn1_biases,
            ffn2_weights,
            ffn2_biases,
            cache_kvs,
            'pre_layer_norm',
            pre_layer_norm,
            'epsilon',
            epsilon,
            'residual_alpha',
            residual_alpha,
            'dropout_rate',
            dropout_rate,
            'rotary_emb_dims',
            rotary_emb_dims,
            'is_test',
            not training,
            'dropout_implementation',
            mode,
            'act_method',
            activation,
            'trans_qkvw',
            trans_qkvw,
            'ring_id',
            ring_id,
            'norm_type',
            norm_type,
            'use_neox_rotary_style',
            use_neox_rotary_style,
            'gqa_group_size',
            gqa_group_size,
        )
        if cache_kvs is not None:
            return final_out, cache_kv_out
        return final_out
    else:
        helper = LayerHelper('fused_multi_transformer', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(
            x, 'x', ['uint16', 'float16'], 'fused_multi_transformer'
        )
        check_dtype(
            dtype,
            'dtype',
            ['uint16', 'float16'],
            'fused_multi_transformer',
        )

        # set inputs
        inputs = {}
        inputs['X'] = [x]
        inputs['LnScale'] = ln_scales
        inputs['QKVW'] = qkv_weights

        if ln_biases is not None:
            inputs['LnBias'] = ln_biases
        if qkv_biases is not None:
            inputs['QKVBias'] = qkv_biases
        if cache_kvs is not None:
            assert len(cache_kvs) == len(qkv_weights)
            inputs['CacheKV'] = cache_kvs
            if time_step is not None:
                inputs['TimeStep'] = time_step
        if pre_caches is not None:
            inputs['PreCaches'] = pre_caches
        if beam_offset is not None:
            inputs['BeamCacheOffset'] = beam_offset
        if rotary_emb_dims > 0:
            inputs['RotaryPosEmb'] = rotary_embs
        inputs['SeqLengths'] = seq_lens
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = linear_weights
        if linear_biases is not None:
            inputs['OutLinearBias'] = linear_biases

        inputs['FFNLnScale'] = ffn_ln_scales
        if ffn_ln_biases is not None:
            inputs['FFNLnBias'] = ffn_ln_biases
        inputs['FFN1Weight'] = ffn1_weights
        if ffn1_biases is not None:
            inputs['FFN1Bias'] = ffn1_biases
        inputs['FFN2Weight'] = ffn2_weights
        if ffn2_biases is not None:
            inputs['FFN2Bias'] = ffn2_biases

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': epsilon,
            'residual_alpha': residual_alpha,
            'dropout_rate': dropout_rate,
            'rotary_emb_dims': rotary_emb_dims,
            'is_test': not training,
            'dropout_implementation': mode,
            'act_method': activation,
            'trans_qkvw': trans_qkvw,
            'ring_id': ring_id,
            'norm_type': norm_type,
            'use_neox_rotary_style': use_neox_rotary_style,
            'gqa_group_size': gqa_group_size,
        }

        outputs = {}
        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        outputs['Out'] = final_out
        if cache_kvs:
            # NOTE: inplace
            outputs['CacheKVOut'] = cache_kvs

        helper.append_op(
            type='fused_multi_transformer',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return (final_out, cache_kvs) if cache_kvs else final_out
