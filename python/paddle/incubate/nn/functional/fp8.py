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

from typing import TYPE_CHECKING

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor


def fused_stack_transpose_quant(
    x: Sequence[Tensor], transpose: bool = True
) -> tuple[Tensor, Tensor]:
    r"""
    Fused operation that performs stacking, optional transposition, and quantization
    on a list of bfloat16 tensors.

    This API supports both dynamic and static graph modes. In dynamic mode, it invokes
    the corresponding C++ core op. In static mode, it appends the op manually to the graph.

    Args:
        x (list[Tensor] or tuple[Tensor]): A list or tuple of bfloat16 tensors, where each tensor
            has shape `[M, N]`. All tensors should have the same shape and dtype.
        transpose (bool, optional): If True, applies a transpose before quantization.
            Default is False.

    Returns:
        tuple:
            - out (Tensor): The quantized output tensor with dtype `float8_e4m3fn`.
            - scale (Tensor): A float32 tensor representing the quantization scale.

    Raises:
        TypeError: If `x` is not a list or tuple of bfloat16 tensors.
        TypeError: If `transpose` is not a boolean.
        RuntimeError: If not running in dynamic mode but trying to call the dynamic op directly.

    Examples:
        .. code-block:: python

            import paddle.incubate.nn.functional as F

            x_vec = []
            num_experts = 1
            seq_len = 2048
            hidden_size = 128
            for _ in range(num_experts):
                x = paddle.randn([seq_len, hidden_size], dtype='bfloat16')
                x = paddle.clip(x, min=-50, max=50)
                x_vec.append(x)

            out, scale = F.fused_stack_transpose_quant(x_vec, transpose=True)

            print(out.shape) # [128, 2048]
            print(scale.shape) # [1, 16]

            out, scale = F.fused_stack_transpose_quant(x_vec, transpose=False)

            print(out.shape) # [2048, 128]
            print(scale.shape) # [16, 1]


    """
    if in_dynamic_or_pir_mode():
        if transpose:
            return _C_ops.fused_stack_transpose_quant(x)
        else:
            return _C_ops.fused_stack_quant(x)


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
        >>> import paddle
        >>> paddle.seed(2023)
        >>> x = paddle.randn([256, 128], dtype='bfloat16')
        >>> x = paddle.clip(x, min=-10, max=10)
        >>> tokens_per_expert = [128, 128]
        >>> outs, scales = paddle.incubate.nn.functional.fused_transpose_split_quant(x, tokens_per_expert)
        >>> print(len(outs))
        2
        >>> print(outs[0].shape)
        [128, 128]
        >>> print(scales[0].shape)
        [1]
        >>> print(outs[0].dtype)
        paddle.float8_e4m3fn
    """

    tokens_per_expert = [int(t) for t in tokens_per_expert]

    if x.shape[0] == 0 or x.shape[1] == 0:
        return [], []

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_transpose_split_quant(
            x, tokens_per_expert, pow_2_scales
        )
