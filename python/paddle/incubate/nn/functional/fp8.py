# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import functools

from paddle import Tensor, _C_ops
from paddle.framework import in_dynamic_or_pir_mode


# special re-use of empty to reduce launch cost.
@functools.cache
def _empty_tensor() -> Tensor:
    """Get tensor with no entries and no data"""
    return Tensor()


def fused_transpose_split_quant(
    x, input_scales, tokens_per_expert, pow_2_scales=False
):
    """
    Applies fused transpose, split, and quantization operation for Mixture of Experts (MoE) models.

    Note:
        This function performs three operations in a single optimized CUDA kernel:
        1. Quantizes input from bfloat16 to float8_e4m3fn format using column-wise scaling
        2. Transposes the matrix from [M, K] to [K, M] layout
        3. Splits the transposed data across multiple experts based on token distribution

    Args:
        x (Tensor): Input tensor of shape [M, K] with dtype bfloat16, where M is the total
            number of tokens and K is the feature dimension. M must be divisible by 128
            for optimal performance.
        tokens_per_expert (List[int]): List containing the number of tokens assigned to each expert.
            Each value should be a multiple of 128 for optimal performance.
            The sum should equal M (total tokens). Values can be 0 for
            unused experts.
        pow_2_scales (bool, optional): Whether to constrain quantization scales to powers of 2
            for better hardware efficiency. If True, scales will be
            rounded to the nearest power of 2. Default: False.

    Returns:
        tuple:
            - outs (List[Tensor]). List of quantized and transposed output tensors, one per expert.
              Each tensor has shape [K, tokens_per_expert[i]] and dtype float8_e4m3fn.
              Empty tensors are included for experts with 0 tokens.
            - scales (List[Tensor]). List of dequantization scale tensors, one per expert.
              Each tensor has shape [K // 128, tokens_per_expert[i] // 128]
              and dtype float32. These are the reciprocal of quantization scales.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> import paddle.incubate.nn.functional as F
            >>> paddle.set_device('gpu')

            >>> x = paddle.randn([384, 512], dtype='bfloat16')
            >>> x = paddle.clip(x, min=-50, max=50)
            >>> tokens_per_expert = [128, 128, 128]
            >>> outs, scales = F.fused_transpose_split_quant(x,None, tokens_per_expert, pow_2_scales=True)
            >>> print(outs[0].shape)
            [512, 128]
            >>> print(scales[0].shape)
            [1, 512]
    """
    tokens_per_expert = [int(t) for t in tokens_per_expert]

    if x.shape[0] == 0 or x.shape[1] == 0:
        return [], []

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_transpose_split_quant(
            x, input_scales, tokens_per_expert, pow_2_scales
        )
