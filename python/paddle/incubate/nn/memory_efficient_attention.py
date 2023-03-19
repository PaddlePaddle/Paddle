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

# The following codes are from https://github.com/facebookresearch/xformers

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import paddle

from .attn_bias import (
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)

SUPPORTED_ATTN_BIAS_TYPES = {
    type(None),
    paddle.Tensor,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    BlockDiagonalMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
}


def _get_seqlen_info(attn_bias):
    if isinstance(
        attn_bias,
        (BlockDiagonalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask),
    ):
        return (
            attn_bias.k_seqinfo.seqstart,
            attn_bias.q_seqinfo.seqstart,
            attn_bias.q_seqinfo.max_seqlen,
            attn_bias.k_seqinfo.max_seqlen,
        )
    else:
        return None, None, -1, -1


def _get_tensor_bias(attn_bias):
    if isinstance(attn_bias, paddle.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    else:
        return None


def memory_efficient_attention(
    query, key, value, attn_bias, p=0.0, scale=None, training=True
):
    assert type(attn_bias) in SUPPORTED_ATTN_BIAS_TYPES
    causal = isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )
    seqstart_k, seqstart_q, max_seqlen_q, _ = _get_seqlen_info(attn_bias)
    # NOTE: compute_logsumexp = training
    is_test = not training
    causal_diagonal = (
        attn_bias.causal_diagonal
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else None
    )
    seqlen_k = (
        attn_bias.k_seqinfo.seqlen
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else None
    )
    attn_bias = _get_tensor_bias(attn_bias)
    # TODO(zhangdanyang): add C++ codes here
