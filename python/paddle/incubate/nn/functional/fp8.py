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
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, _C_ops
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from collections.abc import Sequence


# special re-use of empty to reduce launch cost.
@functools.cache
def _empty_tensor() -> Tensor:
    """Get tensor with no entries and no data"""
    return Tensor()


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
    out: Tensor | None = None,
    bias: Tensor | None = None,
    accumulate: bool = False,
    use_split_accumulator: bool = True,
    is_a_1d_scaled: bool = True,
    is_b_1d_scaled: bool = True,
):

    assert bias is None, "Bias is not supported"

    if bias is None:
        bias = _empty_tensor()
    else:
        assert bias.dtype in (
            paddle.float16,
            paddle.bfloat16,
        ), "Only fp16 and bfloat16 bias are supported."

    M, K = a.shape
    N, K_b = b.shape

    if out is None:
        out = paddle.empty((M, N), dtype=out_dtype)
    else:
        assert out.shape == [
            M,
            N,
        ], f"Expected shape {(M, N)}, got {out.shape}"
        assert out.is_contiguous(), "Output tensor is not contiguous."

    if in_dynamic_or_pir_mode():
        # Create workspace tensor for cuBLAS
        workspace_size = (
            33_554_432
            if paddle.device.cuda.get_device_properties().major >= 9
            else 4_194_304
        )
        workspace = paddle.empty([workspace_size], dtype=paddle.uint8)
        transa, transb = True, False
        grad = False
        math_sm_count = 112

        # Call the C++ operator - it returns (output, pre_gelu_out, workspace_out)
        output, _, _ = _C_ops.fp8_gemm_blockwise_(
            b,
            b_decode_scale,
            a,
            a_decode_scale,
            out,
            bias,
            _empty_tensor(),
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
        return output


def fp8_quant_blockwise(
    x: Tensor,
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
        x_fp8, scale, x_fp8_t, scale_t = _C_ops.fp8_quant_blockwise(
            x,
            epsilon,
            using_1x128,
            input_transpose,
            output_scale_transpose,
            using_e5m2,
            using_pow2_scale,
        )
        # Aligned with kitchen's logic
        if not input_transpose:
            return x_fp8, scale
        else:
            return x_fp8, scale, x_fp8_t, scale_t
