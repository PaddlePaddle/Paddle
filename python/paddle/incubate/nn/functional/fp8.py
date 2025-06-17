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


def fused_act_dequant(
    x: Tensor,
    x_scale: Tensor,
) -> Tensor:
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_act_dequant(x, x_scale)


def fused_swiglu_weighted_bwd(
    o1: Tensor,
    do2_s: Tensor,
    unzipped_probs: Tensor,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)


def fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=False):

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
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_weighted_swiglu_act_quant(
            x, prob, using_pow2_scaling
        )


def fp8_gemm_blockwise(
    a,
    a_decode_scale,
    b,
    b_decode_scale,
    out_dtype,
    bias=None,
    transa=True,
    transb=False,
    grad=False,
    accumulate=False,
    use_split_accumulator=True,
    math_sm_count=112,
    is_a_1d_scaled=True,
    is_b_1d_scaled=True,
):
    """
    Performs FP8 GEMM operation with block-wise scaling using cuBLAS.

    This function computes the FP8 GEMM operation `y = a * b.T + bias` with
    sub-channel (block-wise) quantization scaling support.

    The blockwise scaling granularity is determined by `is_a_1d_scaled` and `is_b_1d_scaled`.

    Supported configurations:
    - 1Dx2D: A is 1D block-wise (1x128), B is 2D block-wise (128x128)
    - 1Dx1D: Both A and B are 1D block-wise (1x128)
    - 2Dx1D: A is 2D block-wise (128x128), B is 1D block-wise (1x128)

    Args:
        a (Tensor): Input tensor A with shape [M, K] and FP8 dtype (float8_e4m3fn or float8_e5m2).
        a_decode_scale (Tensor): Decode scale tensor for A with dtype float32.
                                Scale layout depends on is_a_1d_scaled configuration.
        b (Tensor): Input tensor B with shape [N, K] and FP8 dtype (float8_e4m3fn or float8_e5m2).
        b_decode_scale (Tensor): Decode scale tensor for B with dtype float32.
                                Scale layout depends on is_b_1d_scaled configuration.
        out_dtype (paddle.dtype): Output data type (bfloat16 or float32).
        bias (Tensor, optional): Bias tensor. Currently not supported, must be None. Default: None.
        transa (bool, optional): Whether to transpose tensor A. Default: True.
        transb (bool, optional): Whether to transpose tensor B. Default: False.
        grad (bool, optional): Whether this is a gradient computation. Default: False.
        accumulate (bool, optional): Whether to accumulate into output tensor. Default: False.
        use_split_accumulator (bool, optional): Whether to use split accumulator. Must be True. Default: True.
        math_sm_count (int, optional): Number of SMs to use for math operations. Default: 112.
        is_a_1d_scaled (bool, optional): Whether A uses 1D scaling (1x128 blocks). Default: True.
        is_b_1d_scaled (bool, optional): Whether B uses 1D scaling (1x128 blocks). Default: True.

    Returns:
        Tensor: Output tensor with shape [M, N] and dtype specified by out_dtype.

    Raises:
        TypeError: If input tensors have wrong dtypes.
        ValueError: If tensor shapes don't match or unsupported configurations.
        RuntimeError: If CUDA version or cuBLAS version requirements are not met.
            import paddle

            # Create FP8 input tensors
            M, N, K = 1024, 2048, 512
            a = paddle.randn([M, K], dtype='float32').cast('float8_e4m3fn')
            b = paddle.randn([N, K], dtype='float32').cast('float8_e4m3fn')

            # Create scaling tensors for 1Dx1D configuration
            a_scale = paddle.ones([K//128, M], dtype='float32')
            b_scale = paddle.ones([K//128, N], dtype='float32')

            # Perform FP8 GEMM
            output = paddle.nn.functional.fp8_gemm_blockwise(
                a=a,
                a_decode_scale=a_scale,
                b=b,
                b_decode_scale=b_scale,
                out_dtype=paddle.bfloat16,
                use_split_accumulator=True,
                is_a_1d_scaled=True,
                is_b_1d_scaled=True
            )

            print(f"Output shape: {output.shape}")  # [1024, 2048]
            print(f"Output dtype: {output.dtype}")  # bfloat16

    Note:
        - Requires CUDA 12.6+ and cuBLAS 12.8.4+ for sub-channel scaling support
        - Split accumulator is always required and enabled
        - Bias is currently not supported
        - Only TN layout (transa=True, transb=False) is supported
    """

    # Check output dtype
    if out_dtype not in (paddle.bfloat16, paddle.float32):
        raise TypeError(
            f"out_dtype must be bfloat16 or float32, but got {out_dtype}"
        )
    # Check bias support
    if bias is not None:
        raise ValueError("Bias is currently not supported")

    if in_dynamic_or_pir_mode():
        # Create workspace tensor for cuBLAS
        workspace_size = (
            33_554_432
            if paddle.device.cuda.get_device_properties().major >= 9
            else 4_194_304
        )
        workspace = paddle.empty([workspace_size], dtype=paddle.uint8)

        # Create empty bias and pre_gelu_out tensors
        empty_bias = paddle.empty([0], dtype=paddle.float32)
        empty_pre_gelu_out = paddle.empty([0], dtype=paddle.float32)

        # Call the C++ operator - it returns (out, pre_gelu_out, workspace_out)
        out, _, _ = _C_ops.fp8_gemm_blockwise(
            b,
            b_decode_scale,
            a,
            a_decode_scale,
            empty_bias,
            empty_pre_gelu_out,
            workspace,
            transa,
            transb,
            grad,
            accumulate,
            use_split_accumulator,
            math_sm_count,
            is_b_1d_scaled,
            is_a_1d_scaled,
        )
        return out


def fp8_quant_blockwise(
    X: Tensor,
    epsilon: float = 0.0,
    input_transpose: bool = False,
    output_scale_transpose: bool = True,
    using_pow2_scale: bool = True,
    quant_method: str = "1x128",
    output_type: str = "e4m3",
    name: str | None = None,
):
    if quant_method == "1x128":
        using_1x128 = True
    elif quant_method == "128x128":
        using_1x128 = False
    else:
        raise ValueError("Unsupported quantization method")

    if output_type == "e4m3":
        using_e5m2 = False
    else:
        raise ValueError("Unsupported output type")

    if in_dynamic_or_pir_mode():
        X_fp8, scale, X_fp8_t, scale_t = _C_ops.fp8_quant_blockwise(
            X,
            epsilon,
            using_1x128,
            input_transpose,
            output_scale_transpose,
            using_e5m2,
            using_pow2_scale,
        )
        # Aligned with kitchen's logic
        if not input_transpose:
            return X_fp8, scale
        else:
            return X_fp8, scale, X_fp8_t, scale_t
