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

__all__ = ['fp8_gemm_blockwise']


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

    Examples:
        .. code-block:: python

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

    # Input validation
    if not isinstance(a, paddle.Tensor):
        raise TypeError("a must be a Tensor")
    if not isinstance(b, paddle.Tensor):
        raise TypeError("b must be a Tensor")
    if not isinstance(a_decode_scale, paddle.Tensor):
        raise TypeError("a_decode_scale must be a Tensor")
    if not isinstance(b_decode_scale, paddle.Tensor):
        raise TypeError("b_decode_scale must be a Tensor")

    # Check FP8 dtypes
    if a.dtype not in (paddle.float8_e4m3fn, paddle.float8_e5m2):
        raise TypeError(
            f"a.dtype must be float8_e4m3fn or float8_e5m2, but got {a.dtype}"
        )
    if b.dtype not in (paddle.float8_e4m3fn, paddle.float8_e5m2):
        raise TypeError(
            f"b.dtype must be float8_e4m3fn or float8_e5m2, but got {b.dtype}"
        )

    # Check scale dtypes
    if a_decode_scale.dtype != paddle.float32:
        raise TypeError(
            f"a_decode_scale.dtype must be float32, but got {a_decode_scale.dtype}"
        )
    if b_decode_scale.dtype != paddle.float32:
        raise TypeError(
            f"b_decode_scale.dtype must be float32, but got {b_decode_scale.dtype}"
        )

    # Check output dtype
    if out_dtype not in (paddle.bfloat16, paddle.float32):
        raise TypeError(
            f"out_dtype must be bfloat16 or float32, but got {out_dtype}"
        )

    # Check tensor dimensions
    if len(a.shape) != 2:
        raise ValueError(f"a must be 2D tensor, but got {len(a.shape)}D")
    if len(b.shape) != 2:
        raise ValueError(f"b must be 2D tensor, but got {len(b.shape)}D")

    # Check matrix dimensions compatibility
    M, K = a.shape
    N, K_b = b.shape
    if K != K_b:
        raise ValueError(
            f"Matrix dimensions do not match: a.shape[1]={K}, b.shape[1]={K_b}"
        )

    # Check layout support
    if not (transa and not transb):
        raise ValueError(
            "Only TN layout (transa=True, transb=False) is supported"
        )

    # Check split accumulator requirement
    if not use_split_accumulator:
        raise ValueError(
            "Split accumulator is required (use_split_accumulator must be True)"
        )

    # Check bias support
    if bias is not None:
        raise ValueError("Bias is currently not supported")

    # Check scaling configuration
    if not is_a_1d_scaled and not is_b_1d_scaled:
        raise ValueError("2Dx2D scaling is not supported")

    # Handle empty tensors
    if M == 0 or N == 0 or K == 0:
        return paddle.empty([M, N], dtype=out_dtype)

    if in_dygraph_mode():
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
    else:
        helper = LayerHelper("fp8_gemm_blockwise", **locals())

        # Create output tensor
        out = helper.create_variable_for_type_inference(dtype=out_dtype)

        # Create workspace tensor
        workspace_size = (
            33_554_432
            if paddle.device.cuda.get_device_properties().major >= 9
            else 4_194_304
        )
        workspace = helper.create_variable_for_type_inference(
            dtype=paddle.uint8
        )
        workspace.desc.set_shape([workspace_size])

        # Create empty bias and pre_gelu_out tensors
        empty_bias = helper.create_variable_for_type_inference(
            dtype=paddle.float32
        )
        empty_bias.desc.set_shape([0])
        empty_pre_gelu_out = helper.create_variable_for_type_inference(
            dtype=paddle.float32
        )
        empty_pre_gelu_out.desc.set_shape([0])

        # Create output tensors for pre_gelu_out and workspace_out
        pre_gelu_out = helper.create_variable_for_type_inference(
            dtype=paddle.float32
        )
        workspace_out = helper.create_variable_for_type_inference(
            dtype=paddle.uint8
        )

        helper.append_op(
            type="fp8_gemm_blockwise",
            inputs={
                "A": a,
                "A_scale": a_decode_scale,
                "B": b,
                "B_scale": b_decode_scale,
                "bias": empty_bias,
                "pre_gelu": empty_pre_gelu_out,
                "workspace": workspace,
            },
            outputs={
                "out": out,
                "pre_gelu_out": pre_gelu_out,
                "workspace_out": workspace_out,
            },
            attrs={
                "transa": transa,
                "transb": transb,
                "grad": grad,
                "accumulate": accumulate,
                "use_split_accumulator": use_split_accumulator,
                "math_sm_count": math_sm_count,
                "is_A_1d_scaled": is_a_1d_scaled,
                "is_B_1d_scaled": is_b_1d_scaled,
            },
        )

        return out
