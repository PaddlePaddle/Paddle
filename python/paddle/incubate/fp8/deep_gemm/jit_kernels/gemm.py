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

import math
import os
from functools import cache

import paddle

from ..jit import FP8GemmRuntime, build
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
    get_m_alignment_for_contiguous_layout,
    get_num_sms,
)

global_empty_tensor = paddle.empty([0], dtype=paddle.int32)
# Todo: Use default stream to accelerate CPU time. Optimize here if use multistream to launch gemm kernel.
global_stream = paddle.device.current_stream().stream_base.cuda_stream


def is_tma_multicast_legal(
    shape_dim: int,
    block_dim: int,
    num_tma_multicast: int,
    num_sms: int,
    require_divisible: bool = False,
) -> bool:
    divisible = (
        ceil_div(shape_dim, block_dim) % num_tma_multicast == 0
        or not require_divisible
    )
    return divisible and num_sms % num_tma_multicast == 0


def get_swizzle_mode(block_n: int) -> int:
    elem_size = 2
    for mode_bytes in (128, 64, 32):
        if (block_n * elem_size) % mode_bytes == 0:
            return mode_bytes
    return 0


def get_block_n_padding_for_smem_d(block_n: int) -> int:
    # NOTES: padding is for solving bank conflicts, but wastes shared memory space
    elem_size, requirement = 2, (4, 8)
    bank_stride = (block_n * elem_size) // 4
    padding = (requirement[0] - bank_stride) % requirement[1]
    return (
        ((padding + requirement[1]) if padding < 0 else padding) * 4
    ) // elem_size


def get_smem_config(
    num_stages: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int = 128,
    is_fp32_out: bool = False,
    is_wgrad: bool = False,
) -> tuple[int, int, int]:
    assert block_k == 128

    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = get_swizzle_mode(block_n)
    block_n_padding = (
        get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    )

    # NOTES: `scales_b` in a total manner or per-stage manner
    smem_d = block_m * (block_n + block_n_padding) * (4 if is_fp32_out else 2)
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    smem_scales_b_per_stage = (
        ceil_div(block_n * 4, block_k) * block_k if is_wgrad else 0
    )
    smem_scales_b = ceil_div(k, block_k) * 4 if not is_wgrad else 0
    smem_barrier = num_stages * 8 * 2

    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    smem_size += num_stages * smem_scales_b_per_stage
    smem_size += (
        ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    )
    smem_size += smem_barrier

    # Swizzle and padding are not compatible
    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding


@cache
def get_best_configs(
    m: int,
    n: int,
    k: int,
    num_groups: int,
    num_sms: int,
    is_grouped_contiguous: bool = False,
    is_grouped_masked: bool = False,
    is_fp32_out: bool = False,
    is_wgrad: bool = False,
) -> tuple[int, int, int, int, tuple[int, bool], tuple[int, int, int]]:
    if not is_grouped_contiguous:
        block_ms = (
            64,
            128,
        ) + ((256,) if not is_fp32_out else ())
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(),)
    block_ns = tuple(range(16, 129, 8)) + (
        (
            136,
            152,
        )
        if is_wgrad
        else (
            144,
            160,
        )
    )

    # Avoid bank conflicts for FP32 output
    if is_fp32_out:
        block_ns = [x for x in block_ns if x % 16 == 8]

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (
        ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms)
        if bm
        else None
    )
    get_last_wave_util = lambda bm, bn: fix_wave_saturate(
        (ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms
    )

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES: the block sizes cannot be too large, so at least one dim less than 128
        for block_n in filter(lambda bn: block_m <= 128 or bn <= 128, block_ns):
            success = False
            num_waves, best_num_waves = (
                get_num_waves(block_m, block_n),
                get_num_waves(best_block_m, best_block_n),
            )
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= (
                        block_m == best_block_m and block_n < best_block_n
                    )
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= (
                        block_n == best_block_n and block_m < best_block_m
                    )
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= (
                        block_m != best_block_m and block_n > best_block_n
                    )
            best_block_m, best_block_n = (
                (block_m, block_n) if success else (best_block_m, best_block_n)
            )
    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, sm90_capacity = None, None, 232448
    stage_candidates = tuple(
        filter(lambda s: s <= max(k // 128, 1), (8, 7, 6, 5, 4, 3, 2, 1))
    )
    if 128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4:
        # Unrolling both stages and `num_former_iters` will cause large code size
        stage_candidates = tuple(
            filter(lambda s: s <= max(k // 128, 1), (4, 3, 2, 1))
        )
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(
            num_stages,
            k,
            best_block_m,
            best_block_n,
            is_fp32_out=is_fp32_out,
            is_wgrad=is_wgrad,
        )
        if best_smem_config[0] <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Decide the number of TMA multicasts and whether broadcast on A
    best_tma_multicast_config = (1, True)

    # Try to multicast on the larger block side first
    # NOTES: currently, grouped masked GEMM only supports multicast on A and requires the number of blocks in the N-direction to be even
    is_multicast_legal = {
        "A": is_tma_multicast_legal(
            n, best_block_n, 2, num_sms, is_grouped_masked
        ),
        "B": is_tma_multicast_legal(m, best_block_m, 2, num_sms)
        and not is_grouped_masked,
    }
    for i in ("A", "B") if best_block_m > best_block_n else ("B", "A"):
        if m >= 512 and is_multicast_legal[i]:
            best_tma_multicast_config = (2, i == "A")
            break

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(
        ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups,
        num_waves,
    )
    num_min_sms = (
        ceil_div(num_min_sms, best_tma_multicast_config[0])
        * best_tma_multicast_config[0]
    )
    assert num_min_sms <= num_sms

    return (
        num_min_sms,
        best_block_m,
        best_block_n,
        best_num_stages,
        best_tma_multicast_config,
        best_smem_config,
    )


def gemm_fp8_fp8_bf16_nt(
    lhs: tuple[paddle.Tensor, paddle.Tensor],
    rhs: tuple[paddle.Tensor, paddle.Tensor],
    out: paddle.Tensor,
    num_sms: int | None = None,
) -> None:
    """
    Perform a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., strides[1] = 1.
        The strides[0] of LHS and RHS must be a multiple of 16, and the strides[0] of output must be a multiple of 8.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PaddlePaddle operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[n, k]`,
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == [m, ceil_div(k, 128)]
    assert rhs_scales.shape == [ceil_div(n, 128), ceil_div(k, 128)]
    assert (
        lhs.dtype == paddle.float8_e4m3fn and lhs_scales.dtype == paddle.float32
    )
    assert (
        rhs.dtype == paddle.float8_e4m3fn and rhs_scales.dtype == paddle.float32
    )
    assert out.dtype == paddle.bfloat16
    assert lhs.strides[1] == 1 and out.strides[1] == 1 and rhs.strides[1] == 1

    # LHS scales must be transposed for TMA loads, but not for RHS scales
    # NOTES: `get_col_major_tma_aligned_tensor` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # K must be aligned to 128
    aligned_k = ceil_div(k, 128) * 128

    # Auto-tuning with compilation
    if num_sms is None:
        num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = (
        get_best_configs(m, n, k, 1, num_sms)
    )
    if int(os.getenv("DG_JIT_KERNELS_DEBUG", 0)):
        print(
            f"Auto-tuned gemm_fp8_fp8_bf16_nt as num_sms={num_sms}, block_m={block_m}, block_n={block_n}"
        )
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
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        # Runtime arguments
        "SCALES_B": rhs_scales,
        "GROUPED_LAYOUT": global_empty_tensor,
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
    runtime = build("gemm_fp8_fp8_bf16_nt", FP8GemmRuntime, kwargs)
    runtime(**kwargs)
