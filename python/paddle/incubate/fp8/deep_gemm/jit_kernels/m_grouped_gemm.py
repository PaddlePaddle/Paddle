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

# The file has been adapted from DeepSeek DeepGEMM project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

from __future__ import annotations

import os
from functools import reduce

import paddle

from ..jit import FP8GemmRuntime, build
from .gemm import get_best_configs
from .runtime import (
    GemmType,
    make_2d_tma_a_desc,
    make_2d_tma_b_desc,
    make_2d_tma_d_desc,
    make_2d_tma_scales_desc,
)
from .utils import ceil_div, get_col_major_tma_aligned_tensor, get_num_sms

# Todo: Use default stream to accelerate CPU time. Optimize here if use multistream to launch gemm kernel.
global_stream = paddle.device.current_stream().stream_base.cuda_stream


def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
    lhs: tuple[paddle.Tensor, paddle.Tensor],
    rhs: tuple[paddle.Tensor, paddle.Tensor],
    out: paddle.Tensor,
    m_indices: paddle.Tensor,
    num_sms: int | None = None,
) -> None:
    """
    Perform a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PaddlePaddle operations.
        On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
            `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[num_groups, n, k]`,
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `paddle.int`.
            `m_indices[i]` records the group which the i-th row of the LHS belongs to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m_shape = m_indices.shape
    m__ = reduce(lambda x, y: x * y, m_shape)

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == [m, ceil_div(k, 128)]
    assert rhs_scales.shape == [num_groups, ceil_div(n, 128), ceil_div(k, 128)]
    assert (
        lhs.dtype == paddle.float8_e4m3fn and lhs_scales.dtype == paddle.float32
    )
    assert (
        rhs.dtype == paddle.float8_e4m3fn and rhs_scales.dtype == paddle.float32
    )
    assert out.dtype == paddle.bfloat16
    assert m_indices.dtype == paddle.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    if num_sms is None:
        num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = (
        get_best_configs(m, n, k, 1, num_sms, is_grouped_contiguous=True)
    )
    if int(os.getenv("DG_JIT_KERNELS_DEBUG", 0)):
        print(
            f"Auto-tuned m_grouped_gemm_fp8_fp8_bf16_nt_contiguous as num_sms={num_sms}, block_m={block_m}, block_n={block_n}"
        )
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.GroupedContiguous, lhs, m, k, k, block_m, block_k, num_groups
    )
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.GroupedContiguous, rhs, n, k, k, block_n, block_k, num_groups
    )
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.GroupedContiguous,
        out,
        m,
        n,
        n,
        block_m,
        block_n,
        num_groups,
        smem_config[1],
    )
    tensor_map_scales_a = make_2d_tma_scales_desc(
        GemmType.GroupedContiguous,
        lhs_scales,
        m,
        k,
        block_m,
        block_k,
        num_groups,
    )

    kwargs = {
        # Templated arguments
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_GROUPS": num_groups,
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "GEMM_TYPE": GemmType.GroupedContiguous,
        # Runtime arguments
        "SCALES_B": rhs_scales,
        "GROUPED_LAYOUT": m_indices,
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
        "TENSOR_MAP_A": tensor_map_a,
        "TENSOR_MAP_B": tensor_map_b,
        "TENSOR_MAP_SCALES_A": tensor_map_scales_a,
        "TENSOR_MAP_D": tensor_map_d,
        "STREAM": global_stream,
        "DEVICE_INDEX": out.place.gpu_device_id(),
    }

    # Generate, build and run the kernel
    runtime = build("m_grouped_gemm_fp8_fp8_bf16_nt", FP8GemmRuntime, kwargs)
    runtime(**kwargs)


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    lhs: tuple[paddle.Tensor, paddle.Tensor],
    rhs: tuple[paddle.Tensor, paddle.Tensor],
    out: paddle.Tensor,
    masked_m: paddle.Tensor,
    expected_m: int,
    num_sms: int | None = None,
) -> None:
    """
    Perform a grouped GEMM (masked format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PaddlePaddle operations.
        Moreover, this alignment requirement is different with the contiguous-format kernel, as we require that each batch
            should be separately transposed.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.bfloat16`) of shape `[num_groups, m_max, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[num_groups, m_max, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.bfloat16`) of shape `[num_groups, n, k]`.
             The second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[num_groups, m_max, n]`, representing the result.
        masked_m: a tensor of shape `[num_groups]`, `masked_m[i]` records actual rows of the `lhs[i]` matrix to compute
            in the i-th group.
        expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
            correctly setting this value may lead to better performance.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.shape[0]

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == [num_groups, m, ceil_div(k, 128)]
    assert rhs_scales.shape == [num_groups, ceil_div(n, 128), ceil_div(k, 128)]
    assert (
        lhs.dtype == paddle.float8_e4m3fn and lhs_scales.dtype == paddle.float32
    )
    assert (
        rhs.dtype == paddle.float8_e4m3fn and rhs_scales.dtype == paddle.float32
    )
    assert out.dtype == paddle.bfloat16
    assert masked_m.dtype == paddle.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    if num_sms is None:
        num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = (
        get_best_configs(
            expected_m, n, k, num_groups, num_sms, is_grouped_masked=True
        )
    )
    if int(os.getenv("DG_JIT_KERNELS_DEBUG", 0)):
        print(
            f"Auto-tuned m_grouped_gemm_fp8_fp8_bf16_nt_masked as num_sms={num_sms}, block_m={block_m}, block_n={block_n}"
        )
    # Extra checks for TMA store
    if num_groups > 1 and m > block_m:
        assert m % block_m == 0, (
            f"For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})"
        )

    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.GroupedMasked, lhs, m, k, k, block_m, block_k, num_groups
    )
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.GroupedMasked, rhs, n, k, k, block_n, block_k, num_groups
    )
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.GroupedMasked,
        out,
        m,
        n,
        n,
        block_m,
        block_n,
        num_groups,
        smem_config[1],
    )
    tensor_map_scales_a = make_2d_tma_scales_desc(
        GemmType.GroupedMasked, lhs_scales, m, k, block_m, block_k, num_groups
    )

    kwargs = {
        # Templated arguments
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_GROUPS": num_groups,
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "GEMM_TYPE": GemmType.GroupedMasked,
        # Runtime arguments
        "SCALES_B": rhs_scales,
        "GROUPED_LAYOUT": masked_m,
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
        "TENSOR_MAP_A": tensor_map_a,
        "TENSOR_MAP_B": tensor_map_b,
        "TENSOR_MAP_SCALES_A": tensor_map_scales_a,
        "TENSOR_MAP_D": tensor_map_d,
        "STREAM": paddle.device.cuda.current_stream().cuda_stream,
        "DEVICE_INDEX": out.place.gpu_device_id(),
    }

    # Generate, build and run the kernel
    runtime = build("m_grouped_gemm_fp8_fp8_bf16_nt", FP8GemmRuntime, kwargs)
    runtime(**kwargs)
