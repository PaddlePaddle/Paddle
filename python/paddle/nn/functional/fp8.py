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

import paddle
from paddle import _C_ops
from paddle.base.framework import in_dygraph_mode
from paddle.base.layer_helper import LayerHelper

__all__ = ['fused_transpose_split_quant']


def fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=False):
    """
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
                                        the nearest power of 2 for hardware efficiency.
                                        Default: False.

    Returns:
        tuple: A tuple containing two lists:
            - outs (list[Tensor]): List of quantized tensors, one per expert.
                                    Each tensor has shape [K, tokens_per_expert[i]]
                                    and dtype float8_e4m3fn.
            - scales (list[Tensor]): List of dequantization scale tensors, one per expert.
                                    Each tensor has shape [tokens_per_expert[i]//128]
                                    and dtype float32.

    Raises:
        TypeError: If x is not a Tensor, tokens_per_expert is not a list/tuple,
                    or pow_2_scales is not a bool.
        ValueError: If x is not 2D, tokens_per_expert is empty, any token count
                    is negative or not divisible by 128, sum of token counts doesn't
                    match x.shape[0], or K exceeds the maximum limit.

    Examples:
        .. code-block:: python

            import paddle

            # Create input tensor: 384 tokens, 1024 features
            x = paddle.randn([384, 1024], dtype='bfloat16')

            # Define tokens per expert: 3 experts with 128, 256, 0 tokens respectively
            tokens_per_expert = [128, 256, 0]

            # Perform fused operation
            outs, scales = paddle.nn.functional.fused_transpose_split_quant(
                x, tokens_per_expert, pow_2_scales=False
            )

            # outs[0]: [1024, 128] float8_e4m3fn - first expert's quantized data
            # outs[1]: [1024, 256] float8_e4m3fn - second expert's quantized data
            # outs[2]: [1024, 0] float8_e4m3fn   - third expert's quantized data (empty)

            # scales[0]: [1] float32 - scaling factors for first expert (128//128=1 blocks)
            # scales[1]: [2] float32 - scaling factors for second expert (256//128=2 blocks)
            # scales[2]: [0] float32 - scaling factors for third expert (0//128=0 blocks)

            print(f"Expert 0 output shape: {outs[0].shape}")  # [1024, 128]
            print(f"Expert 1 scale shape: {scales[1].shape}") # [2]0
    """
    if not isinstance(x, paddle.Tensor):
        raise TypeError("x must be a Tensor")

    if x.dtype != paddle.bfloat16:
        raise TypeError(f"x.dtype must be bfloat16, but got {x.dtype}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be 2D tensor, but got {len(x.shape)}D")

    if not isinstance(tokens_per_expert, (list, tuple)):
        raise TypeError("tokens_per_expert must be a list or tuple")

    if len(tokens_per_expert) == 0:
        raise ValueError("tokens_per_expert cannot be empty")

    tokens_per_expert = [int(t) for t in tokens_per_expert]

    for i, tokens in enumerate(tokens_per_expert):
        if tokens < 0:
            raise ValueError(
                f"tokens_per_expert[{i}] must be non-negative, but got {tokens}"
            )
        if tokens % 128 != 0:
            raise ValueError(
                f"tokens_per_expert[{i}] must be divisible by 128, but got {tokens}"
            )

    total_tokens = sum(tokens_per_expert)
    if total_tokens != x.shape[0]:
        raise ValueError(
            f"sum(tokens_per_expert) ({total_tokens}) must equal x.shape[0] ({x.shape[0]})"
        )

    K = x.shape[1]
    if K > 65535 * 128:
        raise ValueError(f"x.shape[1] ({K}) must be <= {65535 * 128}")

    if not isinstance(pow_2_scales, bool):
        raise TypeError("pow_2_scales must be a bool")

    if x.shape[0] == 0 or x.shape[1] == 0:
        return [], []

    if in_dygraph_mode():
        return _C_ops.fused_transpose_split_quant(
            x, tokens_per_expert, pow_2_scales
        )
    else:

        helper = LayerHelper("fused_transpose_split_quant", **locals())

        outs = []
        scales = []

        for i, tokens in enumerate(tokens_per_expert):
            # outs[i]: [K, tokens]
            out = helper.create_variable_for_type_inference(
                dtype=paddle.float8_e4m3fn
            )
            outs.append(out)

            # scales[i]: [tokens//128, K]
            scale = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            scales.append(scale)

        helper.append_op(
            type="fused_transpose_split_quant",
            inputs={"x": x},
            outputs={"outs": outs, "scales": scales},
            attrs={
                "tokens_per_expert": tokens_per_expert,
                "pow_2_scales": pow_2_scales,
            },
        )

        return outs, scales
