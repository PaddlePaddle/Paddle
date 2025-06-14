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
    from paddle import Tensor


def fused_act_dequant(
    x: Tensor,
    x_scale: Tensor,
) -> Tensor:
    r"""
    Fused activation and dequantization operation.

    This function performs dequantization on quantized float8_e4m3fn input tensor
    using the provided scales, converting it to bfloat16 format.

    Args:
        x (Tensor): Input quantized tensor with dtype float8_e4m3fn and shape [rows, cols].
        x_scale (Tensor): Scale tensor for dequantization with dtype float32.
            Can be 1D with shape [scale_groups] where scale_groups = (cols + 127) // 128,
            or 2D with shape [rows, scale_groups] for per-row scaling.

    Returns:
        Tensor: Dequantized output tensor with dtype bfloat16 and same shape as input x.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.incubate.nn.functional as F

            # Example 1: Basic usage with 1D scale
            # Create random data and convert to float8_e4m3fn
            x = paddle.randn([512, 1024], dtype='bfloat16')
            # Simulate quantized input by converting to float8_e4m3fn
            x = x.astype('float8_e4m3fn')
            # Create scale tensor: 1024 // 128 = 8 scale groups
            x_scale = paddle.rand([8], dtype='float32')
            out = F.fused_act_dequant(x, x_scale)
            print(f"Input shape: {x.shape}, Output shape: {out.shape}")
            print(f"Input dtype: {x.dtype}, Output dtype: {out.dtype}")

    Note:
        - Input x must be 2D tensor with dtype float8_e4m3fn
        - x_scale must have dtype float32
        - This operator supports column misalignment cases for flexible quantization
        - Columns divisible by 128 provide optimal performance but are not required
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_act_dequant(x, x_scale)


def fused_swiglu_weighted_bwd(
    o1: Tensor,
    do2_s: Tensor,
    unzipped_probs: Tensor,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Fused SwiGLU probability gradient computation for efficient MoE (Mixture of Experts) training.
    This operator computes the backward pass of SwiGLU activation with probability weighting:
    Forward: o2 = SiLU(x1) * x2 * prob
    where SiLU(x) = x * sigmoid(x)
    Args:
        o1 (Tensor): Input tensor containing concatenated gate and up projections.
                     Shape: [..., hidden_size * 2], dtype: bfloat16
        do2_s (Tensor): Gradient of the scaled output tensor.
                        Shape: [..., hidden_size], dtype: bfloat16
        unzipped_probs (Tensor): Probability weights for each sample.
                                 Shape: [...], dtype: float32
                                 Must have same batch dimensions as o1 and do2_s
        name (str, optional): The default value is None. Normally there is no need for user
                              to set this property. For more information, please refer to
                              :ref:`api_guide_Name`.
    Returns:
        tuple: A tuple containing three tensors:
        - **do1** (Tensor): Gradient w.r.t. input o1. Shape: [..., hidden_size * 2], dtype: bfloat16
        - **probs_grad** (Tensor): Gradient w.r.t. probabilities. Shape: [...], dtype: float32
        - **o2_s** (Tensor): Scaled output o2 * prob. Shape: [..., hidden_size], dtype: bfloat16
    Examples:
        .. code-block:: python

            import paddle
            import paddle.incubate.nn.functional as F

            # Example: Basic 2D usage
            batch_size, hidden_size = 8, 2048
            o1 = paddle.randn([batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([batch_size, 1], dtype='float32')
            do1, probs_grad, o2_s = F.fused_swiglu_weighted_bwd(o1, do2_s, probs)
            print(f"Input shape: {o1.shape}, Output do1 shape: {do1.shape}")
            print(f"Probs gradient shape: {probs_grad.shape}")

    Note:
        - This operator is specifically optimized for MoE training scenarios
        - All input tensors must be on the same device (GPU)
        - The operator leverages vectorized CUDA kernels for optimal performance
        - Batch dimensions (all dimensions except the last for o1/do2_s) must match across inputs
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)


def fused_weighted_swiglu_act_quant(
    x: Tensor,
    prob: Tensor | None = None,
    using_pow2_scaling: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Fused weighted SwiGLU activation with quantization.

    This function performs a fused operation that combines weighted SwiGLU activation
    and quantization. The SwiGLU activation is applied to the input tensor x,
    optionally weighted by prob, and then quantized to FP8 format.

    Args:
        x (Tensor): The input tensor. Must be 2D or higher dimensional with bfloat16 dtype.
                   The last dimension must be even (divisible by 2).
        prob (Tensor, optional): Optional probability tensor for weighting. If provided,
                               must be float32 dtype and have shape [batch_size]. Default: None.
        using_pow2_scaling (bool, optional): Whether to use power-of-2 scaling for quantization.
                                           Default: False.
        name (str, optional): Name for the operation (optional, default is None).
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - out (Tensor): The quantized output tensor with FP8_E4M3FN dtype.
                          Shape: [batch_size, last_dim // 2]
            - scale (Tensor): The quantization scale tensor with float32 dtype.
                            Shape: [batch_size, (last_dim // 2 + 127) // 128]

    Examples:
        .. code-block:: python

            import paddle
            import paddle.incubate.nn.functional as F

            # Example 1: Basic usage without probability weighting
            x = paddle.randn([32, 4096], dtype='bfloat16')
            out, scale = F.fused_weighted_swiglu_act_quant(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print(f"Scale shape: {scale.shape}")

            # Example 2: With probability weighting
            x = paddle.randn([16, 2048], dtype='bfloat16')
            prob = paddle.randn([16], dtype='float32')
            out, scale = F.fused_weighted_swiglu_act_quant(x, prob=prob)
            print(f"Output shape: {out.shape}")
            print(f"Scale shape: {scale.shape}")

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_weighted_swiglu_act_quant(
            x, prob, using_pow2_scaling
        )
