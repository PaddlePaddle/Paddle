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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode


def fused_stack_transpose_quant(
    x: Sequence[Tensor], transpose: bool = True
) -> tuple[Tensor, Tensor]:
    """
    Fused operation that performs stacking, optional transposition, and quantization
    on a list of bfloat16 tensors.

    Args:
        x (list[Tensor] or tuple[Tensor]): A list or tuple of bfloat16 tensors, where each tensor
            has shape `[M, K]`. All tensors should have the same shape and dtype.
        transpose (bool, optional): If True, applies a transpose before quantization.
            Default is True.

    Returns:
        tuple:
            - out (Tensor): The quantized output tensor with dtype `float8_e4m3fn`.
            - scale (Tensor): A float32 tensor representing the quantization scale.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> import paddle.incubate.nn.functional as F
            >>> paddle.set_device('gpu')

            >>> x_vec = []
            >>> num_experts = 1
            >>> seq_len = 2048
            >>> hidden_size = 128
            >>> for _ in range(num_experts):
            ...     x = paddle.randn([seq_len, hidden_size], dtype='bfloat16')
            ...     x = paddle.clip(x, min=-50, max=50)
            ...     x_vec.append(x)

            >>> out, scale = F.fused_stack_transpose_quant(x_vec, transpose=True)
            >>> print(out.shape)
            [128, 2048]
            >>> print(scale.shape)
            [1, 16]
    """
    if in_dynamic_or_pir_mode():
        if transpose:
            return _C_ops.fused_stack_transpose_quant(x)
        else:
            return _C_ops.fused_stack_quant(x)


def fused_act_dequant(
    x: Tensor,
    x_scale: Tensor,
) -> Tensor:
    """
    Applies fused activation and dequantization operation to convert float8 quantized data back to bfloat16.

    This function performs dequantization from float8_e4m3fn to bfloat16 format using pre-computed
    scaling factors. The operation uses vectorized memory access and optimized CUDA kernels for
    high-performance inference in quantized neural networks.

    The dequantization formula is: output = input * scale, where scales are applied per 128-column blocks
    to balance quantization quality and computational efficiency.

    Args:
        x (Tensor): Input quantized tensor with dtype float8_e4m3fn and shape [M, N], where M is the
            number of rows and N is the number of columns. This tensor contains the quantized
            activations from previous layers.
        x_scale (Tensor): Dequantization scale tensor with dtype float32 and shape [M, (N + 127) // 128].
            Each scale value corresponds to a 128-column block in the input tensor.
            The scales are the reciprocal of the quantization scales used during the
            forward quantization process.

    Returns:
        Tensor: Dequantized output tensor with dtype bfloat16 and shape [M, N]. The values are
            computed as input * scale for each corresponding 128-column block.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_act_dequant(x, x_scale)


def fused_swiglu_weighted_bwd(
    o1: Tensor,
    do2_s: Tensor,
    unzipped_probs: Tensor,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes gradients for fused weighted SwiGLU activation function in backward pass.

    This function performs the backward propagation for the SwiGLU (Swish-Gated Linear Unit)
    activation with probability weighting. It computes gradients with respect to both the
    input activations and the probability weights, while also recomputing forward pass values
    for memory efficiency. The kernel automatically selects between vectorized and standard
    implementations based on input dimensions.

    Args:
        o1 (Tensor): Forward pass input tensor with dtype bfloat16 and shape
            [..., intermediate_size * 2]. The tensor is split into two halves:
            - Left half [0:intermediate_size]: x1 values (gate inputs)
            - Right half [intermediate_size:]: x2 values (activation inputs)
            This is the same input used in the forward SwiGLU computation.
        do2_s (Tensor): Upstream gradient tensor with dtype bfloat16 and shape
            [..., intermediate_size]. Contains gradients flowing back from
            the next layer, representing ∂L/∂output before probability weighting.
            Each element corresponds to the gradient of one output element.
        unzipped_probs (Tensor): Probability weighting tensor with dtype float32 and
            shape matching the batch dimensions of o1 and do2_s
            [...]. Each probability value was used to weight the
            corresponding row's output in the forward pass.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - do1 (Tensor): Input gradients with dtype bfloat16 and shape
              [..., intermediate_size * 2]. Layout matches o1:
              - [0:intermediate_size]: ∂L/∂x1 (gradients w.r.t. gate inputs)
              - [intermediate_size:]: ∂L/∂x2 (gradients w.r.t. activation inputs)
            - probs_grad (Tensor): Probability gradients with dtype float32 and
              shape [...]. Each element is ∂L/∂prob for the corresponding batch item,
              computed as the sum of (∂L/∂output_i * SwiGLU_output_i) across the
              intermediate dimension.
            - o2_s (Tensor): Recomputed forward output with dtype bfloat16 and
              shape [..., intermediate_size]. Contains SwiGLU(x1, x2) * prob values.
              This avoids storing forward activations, trading computation for memory.

    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_swiglu_weighted_bwd
            >>> paddle.set_device('gpu')

            >>> batch_size, seq_len = 32, 128
            >>> intermediate_size = 2048

            >>> o1 = paddle.randn([batch_size, seq_len, intermediate_size * 2], dtype='bfloat16')
            >>> do2_s = paddle.randn([batch_size, seq_len, intermediate_size], dtype='bfloat16')
            >>> expert_probs = paddle.rand([batch_size, seq_len, 1], dtype='float32')

            >>> do1, probs_grad, o2_s = fused_swiglu_weighted_bwd(o1, do2_s, expert_probs)
            >>> print(do1.shape)
            [32, 128, 4096]
            >>> print(probs_grad.shape)
            [32, 128, 1]
            >>> print(o2_s.shape)
            [32, 128, 2048]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)


def fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=False):
    """
    Applies fused transpose, split, and quantization operation for Mixture of Experts (MoE) models.

    This function performs three operations in a single optimized CUDA kernel:
    1. Quantizes input from bfloat16 to float8_e4m3fn format using column-wise scaling
    2. Transposes the matrix from [M, K] to [K, M] layout
    3. Splits the transposed data across multiple experts based on token distribution

    The quantization uses block-wise (128-row blocks) maximum absolute values to compute
    scaling factors, ensuring optimal utilization of the float8_e4m3fn range.

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
        Tuple[List[Tensor], List[Tensor]]: A tuple containing:
            - outs (List[Tensor]): List of quantized and transposed output tensors, one per expert.
              Each tensor has shape [K, tokens_per_expert[i]] and dtype float8_e4m3fn.
              Empty tensors are included for experts with 0 tokens.
            - scales (List[Tensor]): List of dequantization scale tensors, one per expert.
              Each tensor has shape [K // 128, tokens_per_expert[i] // 128]
              and dtype float32. These are the reciprocal of quantization scales.

    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_transpose_split_quant
            >>> paddle.set_device('gpu')

            >>> x = paddle.randn([384, 512], dtype='bfloat16')
            >>> x = paddle.clip(x, min=-50, max=50)
            >>> tokens_per_expert = [128, 128, 128]
            >>> outs, scales = fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=True)
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
            x, tokens_per_expert, pow_2_scales
        )


def fused_weighted_swiglu_act_quant(
    x: Tensor,
    prob: Tensor | None = None,
    using_pow2_scaling: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Applies fused weighted SwiGLU activation followed by quantization to float8_e4m3fn format.

    This function combines four operations into a single optimized CUDA kernel:
    1. SwiGLU activation: SwiGLU(x1, x2) = SiLU(x1) * x2 = (x1 * sigmoid(x1)) * x2
    2. Probability weighting: multiply by optional probability factors (for dropout/expert selection)
    3. Activation computation: compute final activation values in float32 precision
    4. Quantization: convert results to float8_e4m3fn with computed scaling factors

    The input tensor is split into two halves along the last dimension:
    - Left half [0, cols/2): first input to SwiGLU (gate values)
    - Right half [cols/2, cols): second input to SwiGLU (activation values)

    Quantization is performed per 128-element blocks to balance precision and efficiency.
    The kernel automatically selects between vectorized (cols % 8 == 0) and standard
    implementations for optimal performance.

    Args:
        x (Tensor): Input tensor with dtype bfloat16 and shape [..., cols], where cols
            must be even. The tensor is interpreted as two concatenated matrices:
            gate values [0:cols/2] and activation values [cols/2:cols].
            Typical shapes: [batch_size, sequence_length, hidden_dim] or
            [tokens, expert_dim] in MoE scenarios.
        prob (Tensor, optional): Probability weighting tensor with dtype float32 and
            shape matching x's batch dimensions [...]. Each value
            multiplies the corresponding row's activation output.
            Used for dropout during training or expert selection
            probabilities during MoE inference. If None, no
            probability weighting is applied. Default: None.
        using_pow2_scaling (bool, optional): Whether to use power-of-2 quantization
            scaling for hardware efficiency.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - **out** (Tensor): Quantized activation output with dtype float8_e4m3fn
              and shape [..., cols/2]. Contains the quantized SwiGLU results.
            - **scale** (Tensor): Dequantization scales with dtype float32 and shape
              [..., (cols/2 + 127) // 128]. Each scale corresponds to a 128-element
              block in the output tensor. To dequantize: original_value = quantized_value / scale.

    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_weighted_swiglu_act_quant
            >>> paddle.set_device('gpu')

            >>> batch_size, seq_len, expert_dim = 32, 128, 2048
            >>> x = paddle.randn([batch_size, seq_len, expert_dim], dtype='bfloat16')
            >>> quantized_out, scales = fused_weighted_swiglu_act_quant(x)
            >>> print(x.shape)
            [32, 128, 2048]
            >>> print(quantized_out.shape)
            [4096, 1024]
            >>> print(scales.shape)
            [4096, 8]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_weighted_swiglu_act_quant(
            x, prob, using_pow2_scaling
        )
