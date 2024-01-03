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

import os

import paddle
from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_mode


def fused_dot_product_attention(
    q,
    k,
    v,
    mask,
    scaling_factor,
    dropout_prob,
    is_training,
    is_causal_masking,
    use_workspace_opt=None,
    return_softmax=False,
):
    r"""
    Fused Dot Product Attention. This is a fusion operator to compute scaled dot product attention in transformer
    model architecture. This operator only supports running on Ampere and Hopper GPU and need cudnn version >= 8906.

    Args:
        q (Tensor): The query tensor. The data type is bfloat16, float16.
        k (Tensor): The key tensor. The data type is bfloat16, float16.
        v (Tensor): The value tensor. The data type is bfloat16, float16.
        mask (Tensor, optional): The mask tensor. The data type is int or bool.
        scaling_factor (float): The scaling factor for the attention scores.
        dropout_prob (float): The dropout probability.
        is_training (bool): A flag indicating whether it is in train phrase or not.
        is_causal_masking (bool): A flag indicating whether it is causal masking or not. If True, the mask will be ignored.
        use_workspace_opt (bool, optional): A flag indicating whether to use workspace optimization or not.
            Default: None. When set to None, the code path will be decided based on its internal logic.
            Currently, it's only supported on Hopper GPU and will be ignored on other devices.
        return_softmax (bool, optional): A flag indicating whether to return softmax_output or not. Default: False.


    Returns:
        A Tensor representing the fused dot product attention, has same shape and data type as `q` .

    Warning:
        This API needs to be integrated into `paddle.nn.functional.scaled_dot_product_attention` in the future.

    """

    batch_size = q.shape[0]
    q_seqlen = q.shape[1]
    k_seqlen = k.shape[1]
    mask_shape = [batch_size, 1, q_seqlen, k_seqlen]

    if mask is None or is_causal_masking is True:
        mask = paddle.ones(mask_shape, dtype='int32')
    else:  # mask is not None and is_causal_masking == False
        assert mask.dtype in [
            paddle.int32,
            paddle.bool,
        ], "mask dtype must be int32 or bool"
        assert (
            mask.shape == mask_shape
        ), "mask shape must be [batch_size, 1, q_seqlen, k_seqlen]"
        mask = mask.astype('int32')

    if use_workspace_opt is False:
        os.environ['CUDNN_FUSE_ATTN_USE_WORKSPACE_OPT'] = '0'
    elif use_workspace_opt is True:
        os.environ['CUDNN_FUSE_ATTN_USE_WORKSPACE_OPT'] = '1'

    if in_dynamic_mode():
        out, softmax, _ = _C_ops.fused_dot_product_attention(
            q,
            k,
            v,
            mask,
            scaling_factor,
            dropout_prob,
            is_training,
            is_causal_masking,
        )
        return out if return_softmax is False else (out, softmax)
    else:
        helper = LayerHelper('fused_dot_product_attention', **locals())
        out = helper.create_variable_for_type_inference(dtype=q.dtype)
        softmax_out = helper.create_variable_for_type_inference(
            dtype=q.dtype, stop_gradient=True
        )
        rng_state = helper.create_variable_for_type_inference(
            dtype='int64', stop_gradient=True
        )

        attrs = {
            "scaling_factor": scaling_factor,
            "dropout_probability": dropout_prob,
            "is_training": is_training,
            "is_causal_masking": is_causal_masking,
        }
        helper.append_op(
            type='fused_dot_product_attention',
            inputs={'q': q, 'k': k, 'v': v, 'mask': mask},
            outputs={
                'out': [out],
                'softmax_out': [softmax_out],
                'rng_state': [rng_state],
            },
            attrs=attrs,
        )
        return out if return_softmax is False else (out, softmax_out)
