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

import pytest
from kitchen import ops, quantization_subchannel_block_hybrid
from kitchen.quantization import QParams, ScalingType

import paddle
from paddle.incubate.nn.functional import fp8


def _get_rmse(y_pred, y_true):
    return paddle.sqrt(paddle.mean((y_pred - y_true) ** 2))


# for reference check
def _pypaddle_qgemm_subchannel_1d2d(a, b, a_descales, b_descales):
    assert a.dtype == paddle.float8_e4m3fn
    assert b.dtype == paddle.float8_e4m3fn
    assert a_descales.dtype == paddle.float32
    assert b_descales.dtype == paddle.float32

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1]

    a_scales_m = a_descales.shape[1]
    a_scales_k = a_descales.shape[0]
    b_scales_k = b_descales.shape[1]
    b_scales_n = b_descales.shape[0]

    assert a_scales_m == M
    assert a_scales_k * 128 == K
    assert b_scales_n * 128 == N
    assert b_scales_k * 128 == K

    a = a.to(paddle.float32)
    b = b.to(paddle.float32)

    out = paddle.zeros((M, N), dtype=paddle.float32)

    for i in range(0, M):
        for j in range(0, N, 128):
            for k in range(0, K, 128):
                out[i, j : j + 128] += (
                    (a[i, k : k + 128] @ b[j : j + 128, k : k + 128].t())
                    * a_descales[k // 128, i]
                    * b_descales[j // 128, k // 128]
                )

    return out

    return out


# for reference check
def _pypaddle_qgemm_subchannel_2d1d(a, b, a_descales, b_descales):
    assert a.dtype == paddle.float8_e4m3fn
    assert b.dtype == paddle.float8_e4m3fn
    assert a_descales.dtype == paddle.float32
    assert b_descales.dtype == paddle.float32

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1]

    a_scales_m = a_descales.shape[0]
    a_scales_k = a_descales.shape[1]
    b_scales_k = b_descales.shape[0]
    b_scales_n = b_descales.shape[1]

    assert a_scales_m * 128 == M
    assert a_scales_k * 128 == K
    assert b_scales_n == N
    assert b_scales_k * 128 == K

    a = a.to(paddle.float32)
    b = b.to(paddle.float32)

    out = paddle.zeros((M, N), dtype=paddle.float32)

    for i in range(0, M, 128):
        for j in range(0, N):
            for k in range(0, K, 128):
                out[i : i + 128, j] += (
                    (a[i : i + 128, k : k + 128] @ b[j, k : k + 128].t())
                    * a_descales[i // 128, k // 128]
                    * b_descales[k // 128, j]
                )

    return out


# for reference check
def _pypaddle_qgemm_subchannel_1d1d(a, b, a_descales, b_descales):
    assert a.dtype == paddle.float8_e4m3fn
    assert b.dtype == paddle.float8_e4m3fn
    assert a_descales.dtype == paddle.float32
    assert b_descales.dtype == paddle.float32

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1]

    a_scales_m = a_descales.shape[1]
    a_scales_k = a_descales.shape[0]
    b_scales_k = b_descales.shape[0]
    b_scales_n = b_descales.shape[1]

    assert a_scales_m == M
    assert a_scales_k * 128 == K
    assert b_scales_n == N
    assert b_scales_k * 128 == K

    a = a.to(paddle.float32)
    b = b.to(paddle.float32)

    out = paddle.zeros((M, N), dtype=paddle.float32)

    for i in range(0, M):
        for j in range(0, N):
            for k in range(0, K, 128):
                out[i, j] += (
                    (a[i, k : k + 128] @ b[j, k : k + 128].t())
                    * a_descales[k // 128, i]
                    * b_descales[k // 128, j]
                )

    return out


@pytest.mark.parametrize("out_dtype", [paddle.bfloat16])
def test_1D2D(out_dtype):
    M = 256
    N = 384
    K = 512
    seed = 0
    paddle.seed(seed)

    A = paddle.randn((M, K), dtype=paddle.bfloat16)
    B = paddle.randn((N, K), dtype=paddle.bfloat16)
    A_dtype = paddle.float8_e4m3fn
    B_dtype = paddle.float8_e4m3fn

    A_quant_tile_shape = (1, 128)
    A_qparams = QParams(
        quant_dtype=A_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=A_quant_tile_shape,
    )
    B_quant_tile_shape = (128, 128)
    B_qparams = QParams(
        quant_dtype=B_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=B_quant_tile_shape,
    )
    quantize_op = quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(
        ops.Backend.CUBLAS
    )

    # TODO(lshpku): replace kitchen quant with framework quant
    qresult_A = quantize_op.quantize(A, A_qparams)

    data_B, scale_B = fp8.fp8_quant_blockwise(
        B,
        quant_method="128x128",
        input_transpose=False,
        output_scale_transpose=False,
        using_pow2_scale=False,
    )
    qA, sA, qB, sB = (qresult_A.data, qresult_A.scale, data_B, scale_B)

    precise_D = A @ B.t()

    pypaddle_D = _pypaddle_qgemm_subchannel_1d2d(qA, qB, sA, sB)
    print(
        f"pypaddle_qgemm_subchannel rmse to precise_D: {_get_rmse(pypaddle_D, precise_D)}"
    )

    D = fp8.fp8_gemm_blockwise(
        qB, sB, qA, sA, out_dtype, is_a_1d_scaled=False, is_b_1d_scaled=True
    )
    D = D.t()
    rmse = _get_rmse(D, pypaddle_D)
    print(f"kitchen.ops.fp8_gemm_blockwise rmse to ref: {rmse}")
    assert rmse < 0.06


@pytest.mark.parametrize("out_dtype", [paddle.bfloat16])
def test_2D1D(out_dtype):
    M = 256
    N = 384
    K = 512 * 2
    seed = 0
    paddle.seed(seed)

    A = paddle.randn((M, K), dtype=paddle.bfloat16)
    B = paddle.randn((N, K), dtype=paddle.bfloat16)
    A_dtype = paddle.float8_e4m3fn
    B_dtype = paddle.float8_e4m3fn
    A_quant_tile_shape = (128, 128)
    A_qparams = QParams(
        quant_dtype=A_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=A_quant_tile_shape,
    )
    B_quant_tile_shape = (1, 128)
    B_qparams = QParams(
        quant_dtype=B_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=B_quant_tile_shape,
    )
    quantize_op = quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(
        ops.Backend.CUBLAS
    )

    data_A, scale_A = fp8.fp8_quant_blockwise(
        A,
        quant_method="128x128",
        input_transpose=False,
        output_scale_transpose=False,
        using_pow2_scale=False,
    )
    # TODO(lshpku): replace kitchen quant with framework quant
    qresult_B = quantize_op.quantize(B, B_qparams)
    qA, sA, qB, sB = (
        data_A,
        scale_A,
        qresult_B.data,
        qresult_B.scale,
    )

    precise_D = A @ B.t()

    pypaddle_D = _pypaddle_qgemm_subchannel_2d1d(qA, qB, sA, sB)
    print(
        f"pypaddle_qgemm_subchannel rmse to precise_D: {_get_rmse(pypaddle_D, precise_D)}"
    )

    D = fp8.fp8_gemm_blockwise(
        qA, sA, qB, sB, out_dtype, is_a_1d_scaled=False, is_b_1d_scaled=True
    )
    rmse = _get_rmse(D, pypaddle_D)
    print(f"kitchen.ops.fp8_gemm_blockwise rmse to ref: {rmse}")
    assert rmse < 0.06


@pytest.mark.parametrize("out_dtype", [paddle.bfloat16, paddle.float32])
def test_1D1D(out_dtype):
    M = 256
    N = 384
    K = 512
    seed = 0
    paddle.seed(seed)

    A = paddle.randn((M, K), dtype=paddle.float32)
    B = paddle.randn((N, K), dtype=paddle.float32)
    A_dtype = paddle.float8_e4m3fn
    B_dtype = paddle.float8_e4m3fn

    A_quant_tile_shape = (1, 128)
    A_qparams = QParams(
        quant_dtype=A_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=A_quant_tile_shape,
    )
    B_quant_tile_shape = (1, 128)
    B_qparams = QParams(
        quant_dtype=B_dtype,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        quant_tile_shape=B_quant_tile_shape,
    )
    quantize_op = quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(
        ops.Backend.CUBLAS
    )

    # TODO(lshpku): replace kitchen quant with framework quant
    qresult_A = quantize_op.quantize(A, A_qparams)
    qresult_B = quantize_op.quantize(B, B_qparams)
    qA, sA, qB, sB = (
        qresult_A.data,
        qresult_A.scale.astype("float32"),
        qresult_B.data,
        qresult_B.scale.astype("float32"),
    )

    '''
    assert qA.shape == (M, K)
    assert sA.shape == (K // 128, M)
    assert qB.shape == (N, K)
    assert sB.shape == (K // 128, N)
    '''

    precise_D = A @ B.t()

    print(f"qA: {qA}")
    print(f"sA: {sA}")
    print(f"qB: {qB}")
    print(f"sB: {sB}")
    pypaddle_D = _pypaddle_qgemm_subchannel_1d1d(qA, qB, sA, sB)
    # assert pypaddle_D.shape == (M, N)
    print(f"pypaddle_D rmse to precise_D: {_get_rmse(pypaddle_D, precise_D)}")

    D = fp8.fp8_gemm_blockwise(
        qA, sA, qB, sB, out_dtype, is_a_1d_scaled=True, is_b_1d_scaled=True
    )

    rmse = _get_rmse(D, pypaddle_D)

    assert rmse < 0.06


# test_1D1D(paddle.bfloat16)
print("-------------------")
test_1D2D(paddle.bfloat16)
print("-------------------")
test_2D1D(paddle.bfloat16)
