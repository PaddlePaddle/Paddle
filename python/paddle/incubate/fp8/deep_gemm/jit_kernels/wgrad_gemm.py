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

import paddle

from ..jit import FP8WGradGemmRuntime, build
from .gemm import get_best_configs
from .runtime import (
    GemmType,
    make_2d_tma_a_desc,
    make_2d_tma_b_desc,
    make_2d_tma_d_desc,
    make_2d_tma_scales_desc,
)
from .utils import (
    ceil_div,
    get_col_major_tma_aligned_tensor,
    get_num_sms,
    get_tma_aligned_size,
)


def wgrad_gemm_fp8_fp8_fp32_nt(
    lhs: tuple[paddle.Tensor, paddle.Tensor],
    rhs: tuple[paddle.Tensor, paddle.Tensor],
    out: paddle.Tensor,
    num_sms: int | None = None,
):
    """
    Perform a weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., strides[1] = 1.
        The strides[0] of LHS and RHS must be a multiple of 16, and the strides[0] of output must be a multiple of 4.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensor require a TMA-aligned transposed format.
            If your input does not match the requirement, this function will do a transposing with a set of slow PaddlePaddle operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.bfloat16`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.bfloat16`) of shape `[n, k]`,
             the second element is an FP32 1x128 scaling tensor for RHS of shape `[n, ⌈k / 128⌉]`.
        out: the FP32 output tensor of shape `[m, n]`, which will be accumulated.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and m > 0
    assert lhs_scales.shape == [m, ceil_div(k, 128)] or lhs_scales.shape == [
        ceil_div(k, 128),
        m,
    ]
    assert rhs_scales.shape == [n, ceil_div(k, 128)] or rhs_scales.shape == [
        ceil_div(k, 128),
        n,
    ]
    assert (
        lhs.dtype == paddle.float8_e4m3fn and lhs_scales.dtype == paddle.float32
    )
    assert (
        rhs.dtype == paddle.float8_e4m3fn and rhs_scales.dtype == paddle.float32
    )
    assert out.dtype == paddle.float32
    assert lhs.strides[1] == 1 and out.strides[1] == 1 and rhs.strides[1] == 1

    # LHS and RHS scales must be transposed for TMA load
    # NOTES: `get_col_major_tma_aligned_tensor` may launch a kernel if not processed by previous kernels
    def get_valid_scales(scales: paddle.Tensor, mn: int):
        if scales.shape == [ceil_div(k, 128), mn]:
            # For k-grouped GEMMs
            scales = scales.transpose([1, 0])
            assert get_tma_aligned_size(mn, 4) == scales.strides[1] == mn
        else:
            scales = get_col_major_tma_aligned_tensor(scales)
        return scales

    lhs_scales = get_valid_scales(lhs_scales, m)
    rhs_scales = get_valid_scales(rhs_scales, n)

    # Do nothing if `k` is zero
    if k == 0:
        return

    # K must be aligned to 128
    aligned_k = ceil_div(k, 128) * 128

    # Auto-tuning with compilation
    if num_sms is None:
        num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = (
        get_best_configs(
            m, n, aligned_k, 1, num_sms, is_fp32_out=True, is_wgrad=True
        )
    )
    if int(os.getenv("DG_JIT_KERNELS_DEBUG", 0)):
        print(
            f"Auto-tuned wgrad_gemm_fp8_fp8_fp32_nt as num_sms={num_sms}, block_m={block_m}, block_n={block_n}"
        )
    num_last_stages = ceil_div(k, 128) % num_stages
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.Normal, lhs, m, k, lhs.strides[0], block_m, block_k, 1
    )
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.Normal, rhs, n, k, rhs.strides[0], block_n, block_k, 1
    )
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.Normal,
        out,
        m,
        n,
        out.strides[0],
        block_m,
        block_n,
        1,
        smem_config[1],
    )
    tensor_map_scales_a = make_2d_tma_scales_desc(
        GemmType.Normal, lhs_scales, m, k, block_m, block_k, 1
    )
    tensor_map_scales_b = make_2d_tma_scales_desc(
        GemmType.Normal, rhs_scales, n, k, block_n, block_k, 1
    )

    kwargs = {
        # Templated arguments
        "GEMM_TYPE": GemmType.Normal,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "M": m,
        "N": n,
        "K": aligned_k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "NUM_STAGES": num_stages,
        "NUM_LAST_STAGES": num_last_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        # Runtime arguments
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
        "TENSOR_MAP_A": tensor_map_a,
        "TENSOR_MAP_B": tensor_map_b,
        "TENSOR_MAP_SCALES_A": tensor_map_scales_a,
        "TENSOR_MAP_SCALES_B": tensor_map_scales_b,
        "TENSOR_MAP_D": tensor_map_d,
        "STREAM": paddle.device.current_stream().stream_base.cuda_stream,
        "DEVICE_INDEX": out.place.gpu_device_id(),
    }

    # Generate, build and run the kernel
    runtime = build("wgrad_gemm_fp8_fp8_fp32_nt", FP8WGradGemmRuntime, kwargs)
    runtime(**kwargs)


def k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(
    lhs: tuple[paddle.Tensor, paddle.Tensor],
    rhs: tuple[paddle.Tensor, paddle.Tensor],
    out: paddle.Tensor,
    batch_sizes: list[int],
    num_sms: int | None = None,
):
    """
    Perform a k-grouped weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        This function handles multiple batches with varying k-dimensions, processing each batch sequentially.
        Each batch's LHS, RHS, and output tensors must be contiguous.
        The RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensors require a TMA-aligned transposed format.

    Arguments:
        lhs: The first element is a flattened FP8 tensor (typed `paddle.bfloat16`) containing all batches of LHS data,
                 and the flattened shape is `[sum(m * k for k in batch_sizes)]`, where m is the number of rows.
             The second element is an FP32 scaling tensor for LHS with shape `[⌈k / 128⌉ for k in batch_sizes), m]`,
                 representing the per-128-channel scaling factors.
        rhs: The first element is a flattened FP8 tensor (typed `paddle.bfloat16`) containing all batches of RHS data,
                 and the flattened shape is `[sum(n * k for k in batch_sizes)]`, where n is the number of rows.
             The second element is an FP32 scaling tensor for RHS with shape `[⌈k / 128⌉ for k in batch_sizes), n]`,
                 representing the per-128-channel scaling factors.
        out: The FP32 output tensor of shape [num_batches, m, n], which will be accumulated.
        batch_sizes: A list of integers specifying the k-dimension for each batch.
    """
    lhs, lhs_scales = paddle.view(lhs[0], [-1]), lhs[1]
    rhs, rhs_scales = paddle.view(rhs[0], [-1]), rhs[1]
    num_batches, m, n = out.shape

    lhs_offset, rhs_offset, scales_offset = 0, 0, 0

    for i in range(num_batches):
        k = batch_sizes[i]
        lhs_slice = paddle.view(lhs[lhs_offset : lhs_offset + m * k], (m, k))
        rhs_slice = paddle.view(rhs[rhs_offset : rhs_offset + n * k], (n, k))
        lhs_scales_slice = lhs_scales[
            scales_offset : scales_offset + ceil_div(k, 128)
        ]
        rhs_scales_slice = rhs_scales[
            scales_offset : scales_offset + ceil_div(k, 128)
        ]
        wgrad_gemm_fp8_fp8_fp32_nt(
            (lhs_slice, lhs_scales_slice),
            (rhs_slice, rhs_scales_slice),
            out[i],
            num_sms,
        )

        lhs_offset += m * k
        rhs_offset += n * k
        scales_offset += ceil_div(k, 128)
