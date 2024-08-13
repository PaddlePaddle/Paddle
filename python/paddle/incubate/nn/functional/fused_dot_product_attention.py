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

from __future__ import annotations

from paddle import Tensor, _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def cudnn_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor | None = None,
    cu_seqlen_q: Tensor | None = None,
    cu_seqlen_k: Tensor | None = None,
    scaling_factor: float = 1.0,
    dropout_prob: float = 0.0,
    training: bool = True,
    mask_type: str | None = None,
    bias_type: str | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Fused Dot Product Attention. This is a fusion operator to compute scaled dot product attention in transformer
    model architecture. This operator only supports running on Ampere and Hopper GPU and need cudnn version >= 8906.

    Args:
        q (Tensor): The query tensor. The data type is bfloat16, float16.
        k (Tensor): The key tensor. The data type is bfloat16, float16.
        v (Tensor): The value tensor. The data type is bfloat16, float16.
        bias (Tensor, optional): The bias tensor. The data type needs to be the same as `q`, `k`, `v`.
            The shape of the bias tensor should be [b,1,s,s] or [b,h,s,s] or [1,1,s,s] or [1, h, s, s].
        cu_seqlen_q (Tensor, optional): The cu_seqlen_q tensor. The data type is int32.
        cu_seqlen_k (Tensor, optional): The cu_seqlen_k tensor. The data type is int32.
        scaling_factor (float): The scaling factor for the attention scores.
        dropout_prob (float): The dropout probability.
        training (bool): A flag indicating whether it is in train phrase or not.
        mask_type (str, optional): The mask type. It can be 'none', 'padding', 'causal', 'paddle_causal'. Default is None.
        bias_type (str, optional): The bias type. It can be 'none', 'pre_scale_bias', 'post_scale_bias'. Default is None.


    Returns:
        A Tensor representing the fused dot product attention, has same shape and data type as `q` .

    Note:
        This API is provides the full functionality of the cuDNN flash attention. Such as dbias, padding_causal mask, etc.
    """

    if mask_type is None:
        mask_type = "none"
    if bias_type is None:
        bias_type = "none"
    assert mask_type in [
        'none',
        'padding',
        'causal',
        'paddle_causal',
    ], "mask_type should be 'none', 'padding', 'causal', 'paddle_causal'"
    assert bias_type in [
        'none',
        'pre_scale_bias',
        'post_scale_bias',
    ], "bias_type should be 'none', 'pre_scale_bias', 'post_scale_bias'"

    if in_dynamic_or_pir_mode():
        out, _, _ = _C_ops.fused_dot_product_attention(
            q,
            k,
            v,
            bias,
            cu_seqlen_q,
            cu_seqlen_k,
            scaling_factor,
            dropout_prob,
            training,
            mask_type,
            bias_type,
        )
        return out
    else:
        helper = LayerHelper('fused_dot_product_attention', **locals())
        out = helper.create_variable_for_type_inference(dtype=q.dtype)
        softmax_out = helper.create_variable_for_type_inference(
            dtype="float", stop_gradient=True
        )
        rng_state = helper.create_variable_for_type_inference(
            dtype='int64', stop_gradient=True
        )

        attrs = {
            "scaling_factor": scaling_factor,
            "dropout_probability": dropout_prob,
            "is_training": training,
            "mask_type_str": mask_type,
            "bias_type_str": bias_type,
        }
        helper.append_op(
            type='fused_dot_product_attention',
            inputs={
                'q': q,
                'k': k,
                'v': v,
                'bias': bias,
                'cu_seqlen_q': cu_seqlen_q,
                'cu_seqlen_k': cu_seqlen_k,
            },
            outputs={
                'out': [out],
                'softmax_out': [softmax_out],
                'rng_state': [rng_state],
            },
            attrs=attrs,
        )
        return out


def fused_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scaling_factor: float | None = None,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    Fused Dot Product Attention. This is a fusion operator to compute scaled dot product attention in transformer
    model architecture. This operator only supports running on Ampere and Hopper GPU and need cudnn version >= 8906.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor,optional): A float mask of the same type as query,
                key, value that is added to the attention score.
        dropout_p (float): The dropout probability.
        is_causal (bool): A flag indicating whether it is causal masking or not. If True, the mask will be ignored.
        scaling_factor (float): The scaling factor for the attention scores.
        training (bool): A flag indicating whether it is in train phrase or not.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the fused dot product attention, has same shape and data type as `q` .

    Note:
        This API is designed to be aligned with `nn.functional.scaled_dot_product_attention` API. So the
            arguments are almost the same as `nn.functional.scaled_dot_product_attention`. This difference
            is that `nn.functional.scaled_dot_product_attention` calls the open source flash attention
            (https://github.com/Dao-AILab/flash-attention). While this API calls the kernel implemented
            by cuDNN, which achieves better performance on the latest GPU architectures (Hopper and After).

        The mask is passed as a post scale bias in this API. So this mask can be a arbitrary mask, not just padding mask.
    """

    if is_causal:
        assert attn_mask is None, "mask must be None when is_causal is True"
        mask_type = "causal"
        bias_type = "none"
    elif attn_mask is not None:
        mask_type = "none"
        bias_type = "post_scale_bias"  # pass mask as a post scale bias
    else:
        mask_type = "none"
        bias_type = "none"

    if attn_mask is not None:
        assert (
            attn_mask.dtype == query.dtype
        ), "attn_mask dtype should be the same as qkv dtype"

    cu_seqlen_q = None
    cu_seqlen_k = None
    if scaling_factor is None:
        head_dim = query.shape[3]
        scaling_factor = head_dim**-0.5
    if in_dynamic_or_pir_mode():
        out, _, _ = _C_ops.fused_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            cu_seqlen_q,
            cu_seqlen_k,
            scaling_factor,
            dropout_p,
            training,
            mask_type,
            bias_type,
        )
        return out
    else:
        helper = LayerHelper('fused_dot_product_attention', **locals())
        out = helper.create_variable_for_type_inference(dtype=query.dtype)
        softmax_out = helper.create_variable_for_type_inference(dtype="float")
        rng_state = helper.create_variable_for_type_inference(dtype='int64')

        attrs = {
            "scaling_factor": scaling_factor,
            "dropout_probability": dropout_p,
            "is_training": training,
            "mask_type_str": mask_type,
            "bias_type_str": bias_type,
        }
        helper.append_op(
            type='fused_dot_product_attention',
            inputs={
                'q': query,
                'k': key,
                'v': value,
                'bias': attn_mask,
                'cu_seqlen_q': cu_seqlen_q,
                'cu_seqlen_k': cu_seqlen_k,
            },
            outputs={
                'out': [out],
                'softmax_out': [softmax_out],
                'rng_state': [rng_state],
            },
            attrs=attrs,
        )
        return out
