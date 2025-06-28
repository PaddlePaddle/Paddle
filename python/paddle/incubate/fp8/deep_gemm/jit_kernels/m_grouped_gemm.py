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

import functools
from functools import reduce

import paddle
from paddle import Tensor

from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"',)
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated grouped GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
GemmType::run(out, rhs_scales, grouped_layout,
              m,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""


@functools.lru_cache
def auto_tuning_with_compilation_grouped_gemm_contiguous(
    m, n, k, num_groups, num_sms
):
    global includes, template
    if num_sms is None:
        num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = (
        get_best_configs(m, n, k, 1, num_sms, is_grouped_contiguous=True)
    )
    runtime = jit_tuner.compile_and_tune(
        m,
        n,
        k,
        name="m_grouped_gemm_fp8_fp8_bf16_nt",
        keys={
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "GEMM_TYPE": "GroupedContiguous",
            "K": k,
            "N": n,
            "NUM_GROUPS": num_groups,
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": num_tma_multicast,
        },
        space=(),
        includes=includes,
        arg_defs=(
            ("lhs", paddle.float8_e4m3fn),
            ("lhs_scales", paddle.float32),
            ("rhs", paddle.float8_e4m3fn),
            ("rhs_scales", paddle.float32),
            ("out", paddle.bfloat16),
            ("grouped_layout", paddle.int32),
            ("m", int),
            ("num_groups", int),
            ("stream", paddle.device.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=template,
    )
    return runtime, num_sms, smem_size


def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
    lhs: tuple[Tensor, Tensor],
    rhs: tuple[Tensor, Tensor],
    out: Tensor,
    m_indices: Tensor,
    num_sms=132,
) -> None:
    """
    Do a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow Paddle operations.
    On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
        `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `paddle.int32`.
            `m_indices[i]` records the group which the j-th row of the LHS belong to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
            `-1` in this tensor indicates no RHS matrix selected, the kernel will skip the computation for that aligned block.
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
    assert lhs_scales.shape == [m, (k + 127) // 128]
    assert rhs_scales.shape == [num_groups, (n + 127) // 128, (k + 127) // 128]
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
    global includes, template
    runtime, num_sms, smem_size = (
        auto_tuning_with_compilation_grouped_gemm_contiguous(
            m, n, k, num_groups, num_sms
        )
    )

    args = (
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        out,
        m_indices,
        m,
        num_groups,
        paddle.device.current_stream().stream_base,
        num_sms,
        smem_size,
    )
    runtime(*args)


@functools.lru_cache
def auto_tuning_with_compilation_grouped_gemm_masked(
    m, expected_m, n, k, num_groups, num_sms
):
    # Auto-tuning with compilation
    global includes, template
    if num_sms is None:
        num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = (
        get_best_configs(expected_m, n, k, num_groups, num_sms)
    )

    # Extra checks for TMA store
    if num_groups > 1 and m > block_m:
        assert (
            m % block_m == 0
        ), f"For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})"

    runtime = jit_tuner.compile_and_tune_group_gemm_masked(
        name="m_grouped_gemm_fp8_fp8_bf16_nt",
        keys={
            "N": n,
            "K": k,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "NUM_GROUPS": num_groups,
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": num_tma_multicast,
            "GEMM_TYPE": "GroupedMasked",
        },
        space=(),
        includes=includes,
        arg_defs=(
            ("lhs", paddle.float8_e4m3fn),
            ("lhs_scales", paddle.float32),
            ("rhs", paddle.float8_e4m3fn),
            ("rhs_scales", paddle.float32),
            ("out", paddle.bfloat16),
            ("grouped_layout", paddle.int32),
            ("m", int),
            ("stream", paddle.device.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=template,
    )

    return runtime, num_sms, smem_size


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    lhs: tuple[Tensor, Tensor],
    rhs: tuple[Tensor, Tensor],
    out: Tensor,
    masked_m: Tensor,
    expected_m: int,
    num_sms=132,
) -> None:
    """
    Do a grouped GEMM (masked format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow Paddle operations.
    Moreover, this alignment requirement is different with the contiguous-format kernel, as we require that each batch
        should be separately transposed.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[num_groups, m_max, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[num_groups, m_max, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `paddle.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
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
    masked_m_shape = masked_m.shape
    num_groups___ = reduce(lambda x, y: x * y, masked_m_shape)

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == [num_groups, m, (k + 127) // 128]
    assert rhs_scales.shape == [num_groups, (n + 127) // 128, (k + 127) // 128]
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

    runtime, num_sms, smem_size = (
        auto_tuning_with_compilation_grouped_gemm_masked(
            m, expected_m, n, k, num_groups, num_sms
        )
    )

    args = (
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        out,
        masked_m,
        m,
        paddle.device.current_stream().stream_base,
        num_sms,
        smem_size,
    )

    # Run the kernel
    runtime(*args)
