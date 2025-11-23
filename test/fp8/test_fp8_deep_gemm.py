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

from cuda.bindings import nvrtc

print(f"NVRTC version: {nvrtc.nvrtcVersion()[1:]}")

import random

from deep_gemm import calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from deep_gemm.jit_kernels.utils import get_m_alignment_for_contiguous_layout

import paddle
from paddle import Tensor
from paddle.incubate.fp8 import deep_gemm


def per_token_cast_to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = (
        paddle.nn.functional.pad(x, (0, pad_size), value=0)
        if pad_size > 0
        else x
    )
    x_view = paddle.view(x, (m, -1, 128))
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=2)
    x_amax = paddle.view(x_amax, (m, -1))
    x_amax = paddle.clip(x_amax, min=1e-4)
    scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
    scaled_x_converted = paddle.view(
        scaled_x.astype(paddle.float8_e4m3fn), (m, n + pad_size)
    )[:, :n]

    x_amax_scaled = paddle.view((x_amax / 448.0), (m, -1))

    result = (scaled_x_converted, x_amax_scaled)
    return result


def per_block_cast_to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype
    )
    x_padded[:m, :n] = x
    x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))

    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
    x_amax = paddle.clip(x_amax, min=1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        paddle.view(x_amax / 448.0, (x_view.shape[0], x_view.shape[2]))
    )


def construct(
    m: int, k: int, n: int
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor]:
    x = paddle.randn((m, k), dtype=paddle.bfloat16)
    y = paddle.randn((n, k), dtype=paddle.bfloat16)
    out = paddle.empty((m, n), dtype=paddle.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_contiguous_grouped(
    num_groups: int, expected_m_per_group: int, k: int, n: int
) -> tuple[
    int, tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor, Tensor
]:
    alignment = get_m_alignment_for_contiguous_layout()
    group_ms = [
        int(expected_m_per_group * random.uniform(0.7, 1.3))
        for _ in range(num_groups)
    ]
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])

    x = paddle.randn((m, k), dtype=paddle.bfloat16)
    y = paddle.randn((num_groups, n, k), dtype=paddle.bfloat16)
    m_indices = paddle.empty([m], dtype=paddle.int32)
    out = paddle.empty((m, n), dtype=paddle.bfloat16)
    ref_out = paddle.randn((m, n), dtype=paddle.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end

    ref_out = paddle.where(
        (m_indices == -1).unsqueeze(1), paddle.zeros_like(ref_out), ref_out
    )

    assert m % 4 == 0, f"TMA alignment error: {m}"
    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (
        paddle.empty_like(y, dtype=paddle.float8_e4m3fn),
        paddle.empty(
            (num_groups, ceil_div(n, 128), k // 128), dtype=paddle.float32
        ),
    )
    for i in range(num_groups):
        # y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        y_fp8_0_i, y_fp8_1_i = per_block_cast_to_fp8(y[i])
        paddle.assign(y_fp8_0_i, y_fp8[0][i])
        paddle.assign(y_fp8_1_i, y_fp8[1][i])

    return m, x_fp8, y_fp8, m_indices, out, ref_out


def construct_masked_grouped(
    num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int
) -> tuple[
    tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor, Tensor
]:
    x = paddle.randn((num_groups, max_m, k), dtype=paddle.bfloat16)
    y = paddle.randn((num_groups, n, k), dtype=paddle.bfloat16)
    out = paddle.empty((num_groups, max_m, n), dtype=paddle.bfloat16)
    ref_out = paddle.einsum("gmk,gnk->gmn", x, y)

    assert max_m % 4 == 0, f"TMA alignment error: {max_m}"
    x_fp8 = (
        paddle.empty_like(x, dtype=paddle.float8_e4m3fn),
        paddle.empty((num_groups, max_m, k // 128), dtype=paddle.float32),
    )
    y_fp8 = (
        paddle.empty_like(y, dtype=paddle.float8_e4m3fn),
        paddle.empty(
            (num_groups, ceil_div(n, 128), k // 128), dtype=paddle.float32
        ),
    )
    for i in range(num_groups):
        # x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        # y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        x_fp8_0_i, x_fp8_1_i = per_token_cast_to_fp8(x[i])
        paddle.assign(x_fp8_0_i, x_fp8[0][i])
        paddle.assign(x_fp8_1_i, x_fp8[1][i])
        y_fp8_0_i, y_fp8_1_i = per_block_cast_to_fp8(y[i])
        paddle.assign(y_fp8_0_i, y_fp8[0][i])
        paddle.assign(y_fp8_1_i, y_fp8[1][i])

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    # Construct mask
    masked_m = paddle.empty((num_groups,), dtype=paddle.int32)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m
    return x_fp8, y_fp8, masked_m, out, ref_out


def construct_wgrad(
    m: int, k: int, n: int
) -> tuple[
    tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor, Tensor
]:
    x = paddle.randn((m, k), dtype=paddle.bfloat16)
    y = paddle.randn((n, k), dtype=paddle.bfloat16)
    residual = paddle.randn((m, n), dtype=paddle.float32) * 10
    out = residual.clone()
    ref_out = residual + (
        x.astype(paddle.float32) @ y.astype(paddle.float32).t()
    )

    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = per_token_cast_to_fp8(y)

    # NOTES: please do inplace add on the `out` later
    return x_fp8, y_fp8, residual, out, ref_out


def construct_k_grouped_wgrad(
    m: int, n: int, k_sizes: list[int]
) -> tuple[
    tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor, list[int]
]:
    num_groups, total_k = len(k_sizes), sum(k_sizes)

    x_flat = paddle.empty((m * total_k,), dtype=paddle.bfloat16)
    y_flat = paddle.empty((n * total_k,), dtype=paddle.bfloat16)
    out = paddle.zeros((num_groups, m, n), dtype=paddle.float32)
    ref_out = paddle.zeros((num_groups, m, n), dtype=paddle.float32)

    # Fill tensors with data and compute reference output
    x_offset, y_offset = 0, 0
    for idx, k in enumerate(k_sizes):
        x_chunk = paddle.randn((m, k), dtype=paddle.bfloat16)
        y_chunk = paddle.randn((n, k), dtype=paddle.bfloat16)

        paddle.assign(
            x_chunk.flatten(), output=x_flat[x_offset : x_offset + m * k]
        )
        paddle.assign(
            y_chunk.flatten(), output=y_flat[y_offset : y_offset + n * k]
        )
        ref_out[idx] = (
            x_chunk.astype(paddle.float32) @ y_chunk.astype(paddle.float32).t()
        )

        x_offset += m * k
        y_offset += n * k

    x_fp8_flat = paddle.empty_like(x_flat, dtype=paddle.float8_e4m3fn)
    y_fp8_flat = paddle.empty_like(y_flat, dtype=paddle.float8_e4m3fn)

    total_scale_factors = sum(ceil_div(k, 128) for k in k_sizes)
    x_scales = paddle.empty((total_scale_factors, m), dtype=paddle.float32)
    y_scales = paddle.empty((total_scale_factors, n), dtype=paddle.float32)

    # Cast to FP8 and prepare scale factors
    x_offset, y_offset, scale_offset = 0, 0, 0
    for k in k_sizes:
        x_fp8_chunk, x_scale_chunk = per_token_cast_to_fp8(
            paddle.view(x_flat[x_offset : x_offset + m * k], (m, k))
        )
        y_fp8_chunk, y_scale_chunk = per_token_cast_to_fp8(
            paddle.view(y_flat[y_offset : y_offset + n * k], (n, k))
        )

        paddle.assign(
            x_fp8_chunk.flatten(),
            output=x_fp8_flat[x_offset : x_offset + m * k],
        )
        paddle.assign(
            y_fp8_chunk.flatten(),
            output=y_fp8_flat[y_offset : y_offset + n * k],
        )

        num_scales = ceil_div(k, 128)
        paddle.assign(
            x_scale_chunk.T,
            output=x_scales[scale_offset : scale_offset + num_scales],
        )
        paddle.assign(
            y_scale_chunk.T,
            output=y_scales[scale_offset : scale_offset + num_scales],
        )

        x_offset += m * k
        y_offset += n * k
        scale_offset += num_scales

    return (x_fp8_flat, x_scales), (y_fp8_flat, y_scales), out, ref_out, k_sizes


def test_gemm() -> None:
    print("Testing GEMM:")
    for m in (64, 128, 4096):
        for k, n in [
            (576, 7168),
            (7168, 2112),
            (1536, 24576),
            (512, 32768),
            (16384, 7168),
            (7168, 4096),
            (2048, 7168),
        ]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            print("diff:", diff)
            assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"

    print()


def test_m_grouped_gemm_contiguous() -> None:
    print("Testing grouped contiguous GEMM:")

    for num_groups, expected_m_per_group, k, n in (
        (4, 8192, 7168, 4096),
        (4, 8192, 2048, 7168),
        (8, 4096, 7168, 4096),
        (8, 4096, 2048, 7168),
        (32, 256, 7168, 4096),
        (32, 256, 2048, 7168),
    ):
        # NOTES: we should mask the unfilled part before calculating difference
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(
            num_groups, expected_m_per_group, k, n
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            x_fp8, y_fp8, out, m_indices
        )
        out = paddle.where(
            (m_indices == -1).unsqueeze(1), paddle.zeros_like(out), out
        )
        diff = calc_diff(out, ref_out)
        print("diff:", diff)
        assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"
    print()


def test_m_grouped_gemm_masked() -> None:
    print("Testing grouped masked GEMM:")
    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in (
            (7168, 4096),
            (2048, 7168),
        ):
            # Test correctness
            for i in range(10):
                x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(
                    num_groups, 4096, expected_m_per_group, k, n
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                    x_fp8, y_fp8, out, masked_m, expected_m_per_group
                )
                for j in range(num_groups):
                    diff = calc_diff(
                        out[j, : masked_m[j].item()],
                        ref_out[j, : masked_m[j].item()],
                    )
                    assert diff < 0.001, (
                        f"{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}"
                    )

            # noinspection PyShadowingNames
            # def test_func():
            #     deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)

            # Test performance with fixed shapes
            # noinspection PyUnboundLocalVariable
            # valid_m = masked_m.sum().item()
            # t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
            # print(f" > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | "
            #       f"throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, "
            #       f"{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s")
    print()


def test_wgrad_gemm():
    print("Testing weight gradient GEMM:")

    for k in (4096, 8192):
        for m, n in (
            (7168, 2112),
            (1536, 24576),
            (512, 32768),
            (16384, 7168),
            (7168, 4096),
            (2048, 7168),
        ):
            # Test correctness
            x_fp8, y_fp8, residual, out, ref_out = construct_wgrad(m, k, n)
            deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            print("diff:", diff)
            assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"
    print()


def test_k_grouped_wgrad_gemm():
    print("Testing grouped weight gradient GEMM:")

    for num_groups, base_k in ((4, 4096), (4, 8192), (8, 4096)):
        for m, n in ((7168, 4096), (2048, 7168)):
            # Vary k sizes around base_k
            k_sizes = [
                base_k + random.randint(-1, 1) * 128
                for _ in range(num_groups - 1)
            ]
            k_sizes.append(base_k * num_groups - sum(k_sizes))

            # Test correctness
            x_fp8, y_fp8, out, ref_out, k_sizes = construct_k_grouped_wgrad(
                m, n, k_sizes
            )
            deep_gemm.k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(
                x_fp8, y_fp8, out, k_sizes
            )

            for idx in range(num_groups):
                diff = calc_diff(out[idx], ref_out[idx])
                print("diff:", diff)
                assert diff < 0.001, (
                    f"{num_groups=}, {m=}, {n=}, k={k_sizes[idx]}, batch={idx}, {diff:.5f}"
                )
    print()


if __name__ == "__main__":
    paddle.seed(0)
    random.seed(0)
    print("Library path:")
    print(f" > {deep_gemm.__path__}\n")
    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()

    test_wgrad_gemm()
    test_k_grouped_wgrad_gemm()
