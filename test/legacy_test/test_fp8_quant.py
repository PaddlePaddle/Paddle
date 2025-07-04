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
            scale_in = paddle.repeat_interleave(
                (
                    paddle.repeat_interleave(scale_in, repeats=128, axis=0)
                    if quant_method == "128x128"
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
        ) in itertools.product(
            self.quant_method_options,
            self.input_transpose_options,
            self.output_scale_transpose_options,
            self.using_pow2_scale_options,
            self.return_transpose_only_options,
        ):
            rmse = self.quant_verify_wrapper(
                x,
                quant_method=quant_method,
                input_transpose=input_transpose,
                output_scale_transpose=output_scale_transpose,
                return_transpose_only=return_transpose_only,
                using_pow2_scale=using_pow2_scale,
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


if __name__ == '__main__':
    unittest.main()
