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

import itertools
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import fp8


class TestFP8Quantization(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.m = 32768
        self.n = 7168
        self.x = paddle.randn((self.m, self.n), dtype=paddle.bfloat16)
        self.rmse_threshold = 3e-2
        self.quant_method_options = ["1x128", "128x128"]
        self.input_transpose_options = [True]  # return non-transpose afterall
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def cal_all_rmse(self, x, x_qdq, transposed: bool):
        if transposed:
            diff_squared = (x_qdq.T - x.to(paddle.float32)) ** 2
        else:
            diff_squared = (x_qdq - x.to(paddle.float32)) ** 2
        rmse = paddle.sqrt(paddle.sum(diff_squared) / x.numel())
        return rmse

    def quant_verify_wrapper(
        self,
        x: paddle.Tensor,
        quant_method: str = "1x128",
        input_transpose: bool = False,
        output_scale_transpose: bool = False,
        return_transpose_only: bool = False,
        using_pow2_scale=True,
        using_ue8m0_scale=False,
    ):
        x = x.contiguous()
        x_q_valid = False
        x_t_q_valid = False
        if input_transpose:
            if return_transpose_only:
                x_t_q, scale_t = fp8.fp8_quant_blockwise(
                    x,
                    quant_method=quant_method,
                    input_transpose=input_transpose,
                    output_scale_transpose=output_scale_transpose,
                    using_pow2_scale=using_pow2_scale,
                    return_transpose_only=return_transpose_only,
                    using_ue8m0_scale=using_ue8m0_scale,
                )
                x_t_q_valid = True
            else:
                x_q, scale, x_t_q, scale_t = fp8.fp8_quant_blockwise(
                    x,
                    quant_method=quant_method,
                    input_transpose=input_transpose,
                    output_scale_transpose=output_scale_transpose,
                    using_pow2_scale=using_pow2_scale,
                    return_transpose_only=return_transpose_only,
                    using_ue8m0_scale=using_ue8m0_scale,
                )
                x_t_q_valid = True
                x_q_valid = True

        else:
            x_q, scale = fp8.fp8_quant_blockwise(
                x,
                quant_method=quant_method,
                input_transpose=input_transpose,
                output_scale_transpose=output_scale_transpose,
                using_pow2_scale=using_pow2_scale,
                return_transpose_only=return_transpose_only,
                using_ue8m0_scale=using_ue8m0_scale,
            )
            x_q_valid = True

        valid_test_list = []

        if x_q_valid:
            valid_test_list.append((False, x_q, scale))
        if x_t_q_valid:
            valid_test_list.append((True, x_t_q, scale_t))

        rmse = 0
        for verify_transpose, x_q_in, scale_in in valid_test_list:
            scale_in = scale_in.T if output_scale_transpose else scale_in
            if using_ue8m0_scale:
                # scale_in is int32 tensor packed with 4 float scales.
                # Explicitly cast to int32 to ensure correct unpacking behavior (4 bytes per element)
                # Ensure contiguous memory layout for view operation
                scale_np = np.ascontiguousarray(scale_in.numpy()).astype(
                    np.int32
                )
                # Unpack: (M, N/4) int32 -> (M, N) uint8
                scale_u8 = scale_np.view(np.uint8)
                # Recover scale value: 2^(exponent - 127)
                scale_float = 2.0 ** (scale_u8.astype(np.float32) - 127)
                scale_in = paddle.to_tensor(scale_float)

            scale_in = paddle.repeat_interleave(
                (
                    paddle.repeat_interleave(scale_in, repeats=128, axis=0)
                    if quant_method == "128x128" and not using_ue8m0_scale
                    else scale_in
                ),
                repeats=128,
                axis=1,
            )
            scale_in = scale_in[: x_q_in.shape[0], : x_q_in.shape[1]]
            self.assertEqual(scale_in.shape, x_q_in.shape)
            x_qdq = x_q_in.astype('float32') * scale_in
            rmse = rmse + self.cal_all_rmse(x, x_qdq, verify_transpose) / len(
                valid_test_list
            )
        return rmse

    def eval_all(
        self,
        x: paddle.Tensor,
    ):
        rmses = []
        for (
            quant_method,
            input_transpose,
            output_scale_transpose,
            using_pow2_scale,
            return_transpose_only,
            using_ue8m0_scale,
        ) in itertools.product(
            self.quant_method_options,
            self.input_transpose_options,
            self.output_scale_transpose_options,
            self.using_pow2_scale_options,
            self.return_transpose_only_options,
            self.using_ue8m0_scale_options,
        ):
            rmse = self.quant_verify_wrapper(
                x,
                quant_method=quant_method,
                input_transpose=input_transpose,
                output_scale_transpose=output_scale_transpose,
                return_transpose_only=return_transpose_only,
                using_pow2_scale=using_pow2_scale,
                using_ue8m0_scale=using_ue8m0_scale,
            )
            self.assertLessEqual(rmse, self.rmse_threshold)
            rmses.append(rmse)
        return rmses

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.bfloat16)

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_quantization_consistency(self):
        rmses1 = self.eval_all(self.x)
        rmses2 = self.eval_all(self.x)
        for r1, r2 in zip(rmses1, rmses1):
            self.assertEqual(r1, r2)


class TestFP8QuantizationFP16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 128 * 12
        self.n = 4096
        self.x = paddle.randn((self.m, self.n), dtype=paddle.float16)
        self.rmse_threshold = 3e-2
        self.quant_method_options = ["1x128", "128x128"]
        self.input_transpose_options = [True]  # return non-transpose afterall
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.float16)


class TestFP8QuantizationUnalignedBF16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 80
        self.n = 4096
        self.dtype_options = paddle.bfloat16
        self.quant_method_options = ["1x128"]
        self.rmse_threshold = 3e-2
        self.using_ue8m0_scale_options = [True, False]

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [False]
        self.using_pow2_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)


class TestFP8QuantizationUnalignedFP16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 8184
        self.n = 2560
        self.dtype_options = paddle.float16
        self.quant_method_options = ["1x128"]

        self.rmse_threshold = 3e-2

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.float16)


class TestFP8QuantizatioUnalignedNBF16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 129
        self.n = 508
        self.dtype_options = paddle.bfloat16
        self.quant_method_options = ["1x128"]
        self.rmse_threshold = 3e-2

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.return_transpose_only_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)


# 0 size
class TestFP8QuantizationZeroSizeBF16(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.m = 0
        self.n = 0
        self.dtype_options = paddle.bfloat16
        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

    def test_fp8_quant_zero_size_tensor(self):
        x_q, scale = fp8.fp8_quant_blockwise(
            self.x,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
            using_pow2_scale=False,
            return_transpose_only=False,
            using_ue8m0_scale=False,
        )
        self.assertEqual(x_q.shape, [0, 0])
        self.assertEqual(x_q.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.shape, [0, 0])
        self.assertEqual(scale.dtype, paddle.float32)


if __name__ == '__main__':
    unittest.main()
