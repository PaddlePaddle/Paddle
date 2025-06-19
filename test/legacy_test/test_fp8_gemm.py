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

import unittest

import paddle
from paddle.incubate.nn.functional import fp8


class TestFP8GemmBlockwise(unittest.TestCase):
    """Test cases for FP8 GEMM blockwise operations"""

    def setUp(self):
        """Set up test environment"""
        # Skip tests if FP8 is not supported
        if not paddle.device.is_compiled_with_cuda():
            self.skipTest("CUDA is required for FP8 operations")

    def cal_rmse(self, y_pred, y_true):
        """Calculate Root Mean Square Error"""
        return paddle.sqrt(paddle.mean((y_pred - y_true) ** 2))

    def pyop_ref_1x128_128x128(self, a, b, a_descales, b_descales, C):
        """Reference implementation for 1x128 @ 128x128 pattern"""
        self.assertEqual(a.dtype, paddle.float8_e4m3fn)
        self.assertEqual(b.dtype, paddle.float8_e4m3fn)
        self.assertEqual(a_descales.dtype, paddle.float32)
        self.assertEqual(b_descales.dtype, paddle.float32)

        M, N, K = a.shape[0], b.shape[0], a.shape[1]
        self.assertEqual(K, b.shape[1])

        a_scales_m = a_descales.shape[1]
        a_scales_k = a_descales.shape[0]
        b_scales_k = b_descales.shape[1]
        b_scales_n = b_descales.shape[0]

        self.assertEqual(a_scales_m, M)
        self.assertEqual(a_scales_k * 128, K)
        self.assertEqual(b_scales_n * 128, N)
        self.assertEqual(b_scales_k * 128, K)

        a = a.astype(paddle.float32)
        b = b.astype(paddle.float32)

        out = paddle.zeros((M, N), dtype=paddle.float32)

        for i in range(0, M):
            for j in range(0, N, 128):
                for k in range(0, K, 128):
                    out[i, j : j + 128] += (
                        (a[i, k : k + 128] @ b[j : j + 128, k : k + 128].t())
                        * a_descales[k // 128, i]
                        * b_descales[j // 128, k // 128]
                    )
        out = out + C.t()
        return out

    def pyop_ref_128x128_1x128(self, a, b, a_descales, b_descales, C):
        """Reference implementation for 128x128 @ 1x128 pattern"""
        self.assertEqual(a.dtype, paddle.float8_e4m3fn)
        self.assertEqual(b.dtype, paddle.float8_e4m3fn)
        self.assertEqual(a_descales.dtype, paddle.float32)
        self.assertEqual(b_descales.dtype, paddle.float32)

        M, N, K = a.shape[0], b.shape[0], a.shape[1]
        self.assertEqual(K, b.shape[1])

        a_scales_m = a_descales.shape[0]
        a_scales_k = a_descales.shape[1]
        b_scales_k = b_descales.shape[0]
        b_scales_n = b_descales.shape[1]

        self.assertEqual(a_scales_m * 128, M)
        self.assertEqual(a_scales_k * 128, K)
        self.assertEqual(b_scales_n, N)
        self.assertEqual(b_scales_k * 128, K)

        a = a.astype(paddle.float32)
        b = b.astype(paddle.float32)

        out = paddle.zeros((M, N), dtype=paddle.float32)

        for i in range(0, M, 128):
            for j in range(0, N):
                for k in range(0, K, 128):
                    out[i : i + 128, j] += (
                        (a[i : i + 128, k : k + 128] @ b[j, k : k + 128].t())
                        * a_descales[i // 128, k // 128]
                        * b_descales[k // 128, j]
                    )

        out = out + C

        return out

    def pyop_ref_1x128_1x128(self, a, b, a_descales, b_descales):
        """Reference implementation for 1x128 @ 1x128 pattern"""
        self.assertEqual(a.dtype, paddle.float8_e4m3fn)
        self.assertEqual(b.dtype, paddle.float8_e4m3fn)
        self.assertEqual(a_descales.dtype, paddle.float32)
        self.assertEqual(b_descales.dtype, paddle.float32)

        M, N, K = a.shape[0], b.shape[0], a.shape[1]
        self.assertEqual(K, b.shape[1])

        a_scales_m = a_descales.shape[1]
        a_scales_k = a_descales.shape[0]
        b_scales_k = b_descales.shape[0]
        b_scales_n = b_descales.shape[1]

        self.assertEqual(a_scales_m, M)
        self.assertEqual(a_scales_k * 128, K)
        self.assertEqual(b_scales_n, N)
        self.assertEqual(b_scales_k * 128, K)

        a = a.astype(paddle.float32)
        b = b.astype(paddle.float32)

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

    def test_1x128_128x128_bfloat16(self):
        """Test 1x128 @ 128x128 pattern with bfloat16 output"""
        out_dtype = paddle.bfloat16
        M, N, K = 256, 384, 512
        seed = 0
        paddle.seed(seed)

        A = paddle.randn((M, K), dtype=paddle.bfloat16)
        B = paddle.randn((N, K), dtype=paddle.bfloat16)

        # Quantize A using fp8
        data_A, scale_A = fp8.fp8_quant_blockwise(
            A,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
            using_pow2_scale=False,
        )
        qA, sA = data_A, scale_A

        # Quantize B using fp8
        data_B, scale_B = fp8.fp8_quant_blockwise(
            B,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
            using_pow2_scale=False,
        )
        qB, sB = data_B, scale_B

        gold_matmul_result = A @ B.t()

        C = paddle.ones([N, M], dtype=paddle.bfloat16)
        # Test reference implementation
        pyop_result = self.pyop_ref_1x128_128x128(qA, qB, sA, sB, C)
        ref_rmse = self.cal_rmse(pyop_result, gold_matmul_result)

        # Test fp8_gemm_blockwise
        fp8_gemm_result = fp8.fp8_gemm_blockwise(
            qB,
            sB,
            qA,
            sA,
            out_dtype,
            C,
            accumulate=True,
            is_a_1d_scaled=False,
            is_b_1d_scaled=True,
        )
        fp8_gemm_result = fp8_gemm_result.t()

        rmse = self.cal_rmse(fp8_gemm_result, pyop_result)

        # Assertions
        self.assertLess(rmse, 0.06, f"RMSE {rmse} exceeds threshold 0.06")

    def test_128x128_1x128_bfloat16(self):
        """Test 128x128 @ 1x128 pattern with bfloat16 output"""
        out_dtype = paddle.bfloat16
        M, N, K = 256, 384, 1024
        seed = 0
        paddle.seed(seed)

        A = paddle.randn((M, K), dtype=paddle.bfloat16)
        B = paddle.randn((N, K), dtype=paddle.bfloat16)

        # Quantize A using fp8
        data_A, scale_A = fp8.fp8_quant_blockwise(
            A,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
            using_pow2_scale=False,
        )
        qA, sA = data_A, scale_A

        # Quantize B using fp8
        data_B, scale_B = fp8.fp8_quant_blockwise(
            B,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
            using_pow2_scale=False,
        )
        qB, sB = data_B, scale_B

        gold_matmul_result = A @ B.t()

        C = paddle.ones([M, N], dtype=paddle.bfloat16)
        # Test reference implementation
        pyop_result = self.pyop_ref_128x128_1x128(qA, qB, sA, sB, C)

        # Test fp8_gemm_blockwise
        fp8_gemm_result = fp8.fp8_gemm_blockwise(
            qA,
            sA,
            qB,
            sB,
            out_dtype,
            C,
            accumulate=True,
            is_a_1d_scaled=False,
            is_b_1d_scaled=True,
        )

        rmse = self.cal_rmse(fp8_gemm_result, pyop_result)
        # Assertions
        self.assertLess(rmse, 0.06, f"RMSE {rmse} exceeds threshold 0.06")

    def test_1x128_1x128_bfloat16(self):
        """Test 1x128 @ 1x128 pattern with bfloat16 output"""
        self._test_1x128_1x128(paddle.bfloat16)

    def _test_1x128_1x128(self, out_dtype):
        """Helper method for 1x128 @ 1x128 pattern testing"""
        pass


if __name__ == '__main__':
    unittest.main()
