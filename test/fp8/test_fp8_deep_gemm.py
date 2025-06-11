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

import random

import paddle
from paddle import Tensor
from paddle.incubate.fp8 import deep_gemm
from paddle.incubate.fp8.deep_gemm import (
    calc_diff,
    ceil_div,
    get_col_major_tma_aligned_tensor,
)


def per_token_cast_to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, n = x.shape
    x_view = paddle.view(x, (m, -1, 128))
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=2)
    x_amax = paddle.view(x_amax, (m, -1))
    x_amax = paddle.clip(x_amax, min=1e-4)
    scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
    scaled_x_converted = paddle.view(
        scaled_x.astype(paddle.float8_e4m3fn), (m, n)
    )

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


def construct_grouped(
    num_groups: int, m: int, k: int, n: int, is_masked: bool
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor]:
    x = paddle.randn((num_groups, m, k), dtype=paddle.bfloat16)
    y = paddle.randn((num_groups, n, k), dtype=paddle.bfloat16)
    out = paddle.empty((num_groups, m, n), dtype=paddle.bfloat16)
    ref_out = paddle.einsum("gmk,gnk->gmn", x, y)

    assert m % 4 == 0, f"TMA alignment error: {m}"
    x_fp8 = (
        paddle.empty_like(x, dtype=paddle.float8_e4m3fn),
        paddle.empty((num_groups, m, k // 128), dtype=paddle.float32),
    )
    y_fp8 = (
        paddle.empty_like(y, dtype=paddle.float8_e4m3fn),
        paddle.empty(
            (num_groups, (n + 127) // 128, k // 128), dtype=paddle.float32
        ),
    )
    for i in range(num_groups):
        x_fp8_0_i, x_fp8_1_i = per_token_cast_to_fp8(x[i])
        paddle.assign(x_fp8_0_i, x_fp8[0][i])
        paddle.assign(x_fp8_1_i, x_fp8[1][i])
        y_fp8_0_i, y_fp8_1_i = per_block_cast_to_fp8(y[i])
        paddle.assign(y_fp8_0_i, y_fp8[0][i])
        paddle.assign(y_fp8_1_i, y_fp8[1][i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (
            paddle.view(x_fp8[0], (-1, k)),
            per_token_cast_to_fp8(paddle.view(x, (-1, k)))[1],
        )
        out, ref_out = paddle.view(out, (-1, n)), paddle.view(ref_out, (-1, n))

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print("Testing GEMM:")
    for m in (64, 128, 4096):
        for k, n in [
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
            assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"

    print()


def test_m_grouped_gemm_contiguous() -> None:
    print("Testing grouped contiguous GEMM:")

    for num_groups, m, k, n in (
        (8, 4096, 7168, 4096),
        (8, 4096, 2048, 7168),
        (4, 8192, 2048, 7168),
        (4, 8192, 7168, 4096),
    ):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_grouped(
            num_groups, m, k, n, is_masked=False
        )
        m_indices = paddle.arange(0, num_groups, dtype=paddle.int32)
        m_indices = paddle.flatten(
            paddle.expand(
                paddle.unsqueeze(m_indices, -1), shape=[num_groups, m]
            )
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            x_fp8, y_fp8, out, m_indices
        )
        diff = calc_diff(out, ref_out)
        print("diff:", diff)
        assert diff < 0.001, f"m={m * num_groups}, {k=}, {n=}, {diff:.5f}"
    print()


def test_m_grouped_gemm_masked() -> None:
    print("Testing grouped masked GEMM:")

    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in (
            (7168, 4096),
            (2048, 7168),
        ):
            # Test correctness
            masked_m_candidates = list(
                filter(
                    lambda candidate: candidate <= m,
                    (64, 128, 192, 256, 320, 384),
                )
            )
            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(
                    num_groups, m, k, n, is_masked=True
                )
                masked_m = paddle.empty((num_groups,), dtype=paddle.int32)
                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                masked_m_float = paddle.cast(masked_m, "float32")
                masked_m_mean = paddle.mean(masked_m_float)
                masked_m_mean_int = paddle.cast(masked_m_mean, "int32")
                expected_m = min(int(masked_m_mean_int + 1), m)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                    x_fp8, y_fp8, out, masked_m, expected_m
                )
                for j in range(num_groups):
                    diff = calc_diff(
                        out[j, : masked_m[j].item()],
                        ref_out[j, : masked_m[j].item()],
                    )
                    print("diff:", diff)
                    assert (
                        diff < 0.001
                    ), f"{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}"

    print()


if __name__ == "__main__":
    paddle.seed(0)
    random.seed(0)
    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
