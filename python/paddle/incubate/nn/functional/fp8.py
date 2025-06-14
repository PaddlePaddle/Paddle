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

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode


def fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=False):
    r"""
    Performs fused transpose, split, and quantization operations for Mixture of Experts (MoE) models.
    This function combines three operations into a single fused kernel for better performance:
    1. Transpose the input tensor from [total_tokens, K] to [K, total_tokens]
    2. Split the transposed tensor into multiple chunks based on tokens_per_expert
    3. Quantize each chunk to float8_e4m3fn format with computed scaling factors
    The quantization process computes per-block scaling factors where each block contains
    128 consecutive tokens. The scaling factor is calculated as 448.0 / max_abs_value
    for each block to maximize the utilization of the float8_e4m3fn range.
    Args:
        x (Tensor): Input tensor with shape [total_tokens, K] and dtype bfloat16.
            total_tokens must equal sum(tokens_per_expert).
        tokens_per_expert (list or tuple): List of token counts for each expert.
            Each value must be non-negative and divisible by 128.
        pow_2_scales (bool, optional): If True, quantization scales are rounded to
            the nearest power of 2 for hardware efficiency. Default: False.
    Returns:
        tuple: A tuple containing two lists:
        - outs (list[Tensor]): List of quantized tensors, one per expert.
          Each tensor has shape [K, tokens_per_expert[i]] and dtype float8_e4m3fn.
        - scales (list[Tensor]): List of dequantization scale tensors, one per expert.
          Each tensor has shape [tokens_per_expert[i]//128] and dtype float32.
    Raises:
        TypeError: If x is not a Tensor, tokens_per_expert is not a list/tuple,
            or pow_2_scales is not a bool.
        ValueError: If x is not 2D, tokens_per_expert is empty, any token count
            is negative or not divisible by 128, sum of token counts doesn't
            match x.shape[0], or K exceeds the maximum limit.
    Examples:
        .. code-block:: python

            import paddle
            # Simple example with non-zero tokens
            x = paddle.randn([256, 128], dtype='bfloat16')
            x = paddle.clip(x, min=-10, max=10)
            tokens_per_expert = [128, 128]
            outs, scales = paddle.incubate.nn.functional.fused_transpose_split_quant(x, tokens_per_expert)
            len(outs)  # Number of experts
            outs[0].shape  # First expert output
            scales[0].shape  # First expert scale
            str(outs[0].dtype)
    """

    tokens_per_expert = [int(t) for t in tokens_per_expert]

    if x.shape[0] == 0 or x.shape[1] == 0:
        return [], []

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_transpose_split_quant(
            x, tokens_per_expert, pow_2_scales
        )
