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

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from ...base.data_feeder import check_dtype

if TYPE_CHECKING:
    from paddle import Tensor


def fused_act_dequant(x, x_scale):
    """
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
        >>> import paddle
        >>> import paddle.nn.functional as F

        >>> # Example 1: Basic usage with 1D scale
        >>> x = paddle.randint(0, 255, [512, 1024], dtype='uint8')  # Simulate quantized data
        >>> x = x.astype('float8_e4m3fn')  # Convert to float8_e4m3fn
        >>> x_scale = paddle.rand([8], dtype='float32')  # 1024 // 128 = 8 scale groups
        >>> out = F.fused_act_dequant(x, x_scale)
        >>> print(f"Input shape: {x.shape}, Output shape: {out.shape}")
        >>> print(f"Input dtype: {x.dtype}, Output dtype: {out.dtype}")

        >>> # Example 2: Per-row scaling with 2D scale
        >>> x = paddle.randint(0, 255, [256, 512], dtype='uint8')
        >>> x = x.astype('float8_e4m3fn')
        >>> x_scale = paddle.rand([256, 4], dtype='float32')  # 512 // 128 = 4 scale groups per row
        >>> out = F.fused_act_dequant(x, x_scale)
        >>> print(f"Output shape: {out.shape}, dtype: {out.dtype}")

    Note:
        - Input x must be 2D tensor with dtype float8_e4m3fn
        - x_scale must have dtype float32
        - The number of columns in x should be divisible by 128 for optimal performance
        - Each scale value corresponds to 128 consecutive elements in the column dimension
    """
    # Input validation
    if not isinstance(x, paddle.Tensor):
        raise TypeError("x must be a Tensor")

    if not isinstance(x_scale, paddle.Tensor):
        raise TypeError("x_scale must be a Tensor")

    # Check data types
    if x.dtype != paddle.float8_e4m3fn:
        raise TypeError(f"x.dtype must be float8_e4m3fn, but got {x.dtype}")

    if x_scale.dtype != paddle.float32:
        raise TypeError(
            f"x_scale.dtype must be float32, but got {x_scale.dtype}"
        )

    # Check dimensions
    if len(x.shape) != 2:
        raise ValueError(f"x must be 2D tensor, but got {len(x.shape)}D")

    if len(x_scale.shape) not in [1, 2]:
        raise ValueError(
            f"x_scale must be 1D or 2D tensor, but got {len(x_scale.shape)}D"
        )

    # Get dimensions
    rows, cols = x.shape
    expected_scale_groups = (cols + 127) // 128

    # Validate x_scale shape
    if len(x_scale.shape) == 1:
        if x_scale.shape[0] != expected_scale_groups:
            raise ValueError(
                f"For 1D x_scale, size should be {expected_scale_groups} "
                f"(cols + 127) // 128, but got {x_scale.shape[0]}"
            )
    else:  # 2D
        if x_scale.shape[0] != rows:
            raise ValueError(
                f"For 2D x_scale, first dimension should be {rows} "
                f"(same as x.shape[0]), but got {x_scale.shape[0]}"
            )
        if x_scale.shape[1] != expected_scale_groups:
            raise ValueError(
                f"For 2D x_scale, second dimension should be {expected_scale_groups} "
                f"(cols + 127) // 128, but got {x_scale.shape[1]}"
            )

    # Handle empty tensors
    if rows == 0 or cols == 0:
        return paddle.empty(
            [rows, cols], dtype=paddle.bfloat16, place=x.place()
        )

    # Call the kernel
    if paddle.in_dynamic_mode():
        return _C_ops.fused_act_dequant(x, x_scale)
    else:
        # Static graph mode
        helper = LayerHelper("fused_act_dequant", **locals())

        # Create output variable
        out = helper.create_variable_for_type_inference(dtype=paddle.bfloat16)

        # Append op
        helper.append_op(
            type="fused_act_dequant",
            inputs={"x": x, "x_scale": x_scale},
            outputs={"out": out},
            attrs={},
        )

        return out


def fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs, name=None):
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
            from paddle.incubate.nn.functional import fused_swiglu_probs_grad
            # Example 1: Basic 2D usage
            batch_size, hidden_size = 8, 2048
            o1 = paddle.randn([batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([batch_size], dtype='float32')
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
            # Example 2: 3D tensor (sequence + batch)
            seq_len, batch_size, hidden_size = 512, 8, 2048
            o1 = paddle.randn([seq_len, batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([seq_len, batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([seq_len, batch_size], dtype='float32')
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
            # Example 3: MoE scenario with 4D tensors
            seq_len, top_k, batch_size, hidden_size = 512, 2, 8, 2048
            o1 = paddle.randn([seq_len, top_k, batch_size, hidden_size * 2], dtype='bfloat16')
            do2_s = paddle.randn([seq_len, top_k, batch_size, hidden_size], dtype='bfloat16')
            probs = paddle.rand([seq_len, top_k, batch_size], dtype='float32')
            do1, probs_grad, o2_s = fused_swiglu_probs_grad(o1, do2_s, probs)
    Note:
        - This operator is specifically optimized for MoE training scenarios
        - All input tensors must be on the same device (GPU)
        - The operator leverages vectorized CUDA kernels for optimal performance
        - Batch dimensions (all dimensions except the last for o1/do2_s) must match across inputs
    """

    if not isinstance(o1, paddle.Tensor):
        raise TypeError(f"o1 must be a Tensor, but got {type(o1)}")
    if not isinstance(do2_s, paddle.Tensor):
        raise TypeError(f"do2_s must be a Tensor, but got {type(do2_s)}")
    if not isinstance(unzipped_probs, paddle.Tensor):
        raise TypeError(
            f"unzipped_probs must be a Tensor, but got {type(unzipped_probs)}"
        )

    if o1.dtype != paddle.bfloat16:
        raise ValueError(f"o1 must have dtype bfloat16, but got {o1.dtype}")
    if do2_s.dtype != paddle.bfloat16:
        raise ValueError(
            f"do2_s must have dtype bfloat16, but got {do2_s.dtype}"
        )
    if unzipped_probs.dtype != paddle.float32:
        raise ValueError(
            f"unzipped_probs must have dtype float32, but got {unzipped_probs.dtype}"
        )

    if o1.place != do2_s.place or o1.place != unzipped_probs.place:
        raise ValueError("All input tensors must be on the same device")

    if len(o1.shape) != len(do2_s.shape):
        raise ValueError(
            f"o1 and do2_s must have same number of dimensions, "
            f"but got {len(o1.shape)} vs {len(do2_s.shape)}"
        )

    if o1.shape[-1] != do2_s.shape[-1] * 2:
        raise ValueError(
            f"Last dimension of o1 must be twice that of do2_s, "
            f"but got {o1.shape[-1]} vs {do2_s.shape[-1] * 2}"
        )

    if in_dynamic_mode():
        return _C_ops.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)
    else:

        helper = paddle.static.LayerHelper(
            'fused_swiglu_weighted_bwd', **locals()
        )

        do1 = helper.create_variable_for_type_inference(dtype=o1.dtype)
        probs_grad = helper.create_variable_for_type_inference(
            dtype=paddle.float32
        )
        o2_s = helper.create_variable_for_type_inference(dtype=do2_s.dtype)

        helper.append_op(
            type='fused_swiglu_weighted_bwd',
            inputs={'o1': o1, 'do2_s': do2_s, 'unzipped_probs': unzipped_probs},
            outputs={'do1': do1, 'probs_grad': probs_grad, 'o2_s': o2_s},
            attrs={},
        )

        return do1, probs_grad, o2_s


def fused_weighted_swiglu_act_quant(
    x: Tensor,
    prob: Tensor | None = None,
    using_pow2_scaling: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
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

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # Example 1: Basic usage without probability weighting
            >>> x = paddle.randn([32, 4096], dtype='bfloat16')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(x)
            >>> print(f"Input shape: {x.shape}")
            >>> print(f"Output shape: {out.shape}")
            >>> print(f"Scale shape: {scale.shape}")
            Input shape: [32, 4096]
            Output shape: [32, 2048]
            Scale shape: [32, 16]

            >>> # Example 2: With probability weighting
            >>> x = paddle.randn([16, 2048], dtype='bfloat16')
            >>> prob = paddle.randn([16], dtype='float32')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(x, prob=prob)
            >>> print(f"Output shape: {out.shape}")
            >>> print(f"Scale shape: {scale.shape}")
            Output shape: [16, 1024]
            Scale shape: [16, 8]

            >>> # Example 3: With power-of-2 scaling
            >>> x = paddle.randn([8, 1024], dtype='bfloat16')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(
            ...     x, using_pow2_scaling=True
            ... )
            >>> print(f"Output shape: {out.shape}")
            Output shape: [8, 512]
    """

    def __check_input(x, prob, using_pow2_scaling):
        # Check input tensor x
        check_dtype(
            x.dtype,
            'x',
            ['bfloat16'],
            'fused_weighted_swiglu_act_quant',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The input tensor x must be at least 2-dimensional, "
            f"but received x's dimensional: {len(input_shape)}.\n"
        )

        last_dim = input_shape[-1]
        assert last_dim % 2 == 0, (
            "The last dimension of input tensor x must be even (divisible by 2), "
            f"but got last_dim = {last_dim}.\n"
        )

        # Check probability tensor if provided
        if prob is not None:
            check_dtype(
                prob.dtype,
                'prob',
                ['float32'],
                'fused_weighted_swiglu_act_quant',
            )

            prob_shape = list(prob.shape)
            batch_size = 1
            for i in range(len(input_shape) - 1):
                batch_size *= input_shape[i]

            assert len(prob_shape) == 1 and prob_shape[0] == batch_size, (
                f"The prob tensor must have shape [{batch_size}], "
                f"but got shape {prob_shape}.\n"
            )

        # Check using_pow2_scaling
        assert isinstance(using_pow2_scaling, bool), (
            "The using_pow2_scaling must be a boolean value, "
            f"but got {type(using_pow2_scaling)}.\n"
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_weighted_swiglu_act_quant(
            x, prob, using_pow2_scaling
        )
    else:
        __check_input(x, prob, using_pow2_scaling)

        helper = LayerHelper('fused_weighted_swiglu_act_quant', **locals())
        out = helper.create_variable_for_type_inference(dtype='float8_e4m3fn')
        scale = helper.create_variable_for_type_inference(dtype='float32')

        inputs = {'x': [x]}
        if prob is not None:
            inputs['prob'] = [prob]

        helper.append_op(
            type='fused_weighted_swiglu_act_quant',
            inputs=inputs,
            attrs={'using_pow2_scaling': using_pow2_scaling},
            outputs={'out': [out], 'scale': [scale]},
        )
        return out, scale
