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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import math
from functools import cache, lru_cache

import paddle
from paddle import Tensor

from .tuner import jit_tuner
from .utils import (
    ceil_div,
    get_col_major_tma_aligned_tensor,
    get_m_alignment_for_contiguous_layout,
    get_num_sms,
)

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"',)
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = 128;
constexpr auto BLOCK_N_PADDING = {BLOCK_N_PADDING};
constexpr auto kSwizzleDMode = {SWIZZLE_D_MODE};
constexpr auto kNumGroups = 1;
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};
constexpr auto kIsTMAMulticastOnA = {IS_TMA_MULTICAST_ON_A};

// Make a templated GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, GemmType::Normal>;

// Launch kernel
auto tma_a_desc = gemm_t::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = gemm_t::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = gemm_t::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = gemm_t::make_2d_tma_d_desc(out, m);
gemm_t::run(out, rhs_scales, nullptr,
            m,
            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
            stream, num_sms, smem_size);
"""


def is_tma_multicast_legal(
    shape_dim: int, block_dim: int, num_tma_multicast: int, num_sms: int
) -> bool:
    if num_tma_multicast == 1:
        return True
    return (
        shape_dim % (block_dim * num_tma_multicast) == 0
    ) and num_sms % num_tma_multicast == 0


def get_swizzle_mode(block_n: int) -> int:
    # TODO: remove some candidates if slow
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
    num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128
) -> tuple[int, int, int]:
    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = get_swizzle_mode(block_n)
    block_n_padding = (
        get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    )

    smem_d = block_m * (block_n + block_n_padding) * 2
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    smem_scales_b = ceil_div(k, block_k) * 4
    smem_barrier = num_stages * 8 * 2

    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_b_per_stage
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
) -> tuple[int, int, int, int, tuple[int, bool], tuple[int, int, int]]:
    if not is_grouped_contiguous:
        block_ms = (64, 128, 256)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(),)
    block_ns = (*tuple(range(16, 129, 8)), 144, 160)

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
        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in filter(lambda bn: block_m <= 128 or bn <= 128, block_ns):

            success = False
            num_waves, best_num_waves = get_num_waves(
                block_m, block_n
            ), get_num_waves(best_block_m, best_block_n)
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
    stage_candidates = (8, 7, 6, 5, 4, 3)
    if 128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4:
        # Unrolling both stages and `num_former_iters` will cause large code size
        stage_candidates = (4, 3)
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(
            num_stages, k, best_block_m, best_block_n
        )
        if best_smem_config[0] <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Decide the number of TMA multicast and whether broadcast on A
    best_tma_multicast_config = (1, 1)

    # Try to multicast on the larger block side first
    is_dense_gemm = (not is_grouped_contiguous) and (not is_grouped_masked)
    is_multicast_legal = {
        'A': is_tma_multicast_legal(n, best_block_n, 2, num_sms),
        'B': is_tma_multicast_legal(m, best_block_m, 2, num_sms)
        and is_dense_gemm,
    }
    for i in ('A', 'B') if best_block_m > best_block_n else ('B', 'A'):
        if m >= 512 and is_multicast_legal[i]:
            best_tma_multicast_config = (2, int(i == 'A'))
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


@lru_cache
def auto_tuning_with_compilation(m, n, k, num_sms):
    global includes, template
    if num_sms is None:
        num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = (
        get_best_configs(m, n, k, 1, num_sms)
    )
    runtime = jit_tuner.compile_and_tune(
        m,
        n,
        k,
        name="gemm_fp8_fp8_bf16_nt",
        keys={
            'SWIZZLE_D_MODE': smem_config[1],
            'BLOCK_N_PADDING': smem_config[2],
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "K": k,
            "N": n,
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": tma_multicast_config[0],
            'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1],
        },
        space=(),
        includes=includes,
        arg_defs=(
            ("lhs", paddle.float8_e4m3fn),
            ("lhs_scales", paddle.float32),
            ("rhs", paddle.float8_e4m3fn),
            ("rhs_scales", paddle.float32),
            ("out", paddle.bfloat16),
            ("m", int),
            ("stream", paddle.device.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=template,
    )
    return runtime, num_sms, smem_config


def gemm_fp8_fp8_bf16_nt(
    lhs: tuple[Tensor, Tensor],
    rhs: tuple[Tensor, Tensor],
    out: Tensor,
    num_sms=132,
) -> None:
    """
    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow Paddle operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape
    assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == [m, (k + 127) // 128]
    assert rhs_scales.shape == [(n + 127) // 128, (k + 127) // 128]
    assert (
        lhs.dtype == paddle.float8_e4m3fn and lhs_scales.dtype == paddle.float32
    )
    assert (
        rhs.dtype == paddle.float8_e4m3fn and rhs_scales.dtype == paddle.float32
    )
    assert out.dtype == paddle.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return
    runtime, num_sms, smem_config = auto_tuning_with_compilation(
        m, n, k, num_sms
    )
    args = (
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        out,
        m,
        paddle.device.current_stream().stream_base,
        num_sms,
        smem_config[0],
    )
    # Run the kernel.
    runtime(*args)
