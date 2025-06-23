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


class TestFP8Quantization(unittest.TestCase):

    def setUp(self):
        paddle.seed(42)
        self.m = 32768
        self.n = 7168
        self.x = paddle.randn((self.m, self.n), dtype=paddle.bfloat16)
        self.rmse_threshold = 0.03

    def cal_all_rmse(self, x, x_q, x_qdq, transposed: bool):
        if transposed:
            diff_squared = (x_qdq.T - x.to(paddle.float32)) ** 2
        else:
            diff_squared = (x_qdq - x.to(paddle.float32)) ** 2
        rmse = paddle.sqrt(paddle.sum(diff_squared) / x.numel())
        return rmse

    def eval_per_128x128_quant(
        self,
        x: paddle.Tensor,
        input_transpose: bool = False,
        scale_transpose=False,
    ):
        x = x.contiguous()
        if input_transpose:
            x_q, scale, x_t_q, scale_t = fp8.fp8_quant_blockwise(
                x,
                quant_method="128x128",
                input_transpose=True,
                output_scale_transpose=False,
                using_pow2_scale=False,
            )
        else:
            x_q, scale = fp8.fp8_quant_blockwise(
                x,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
                using_pow2_scale=False,
            )

        self.assertEqual(len(x_q.shape), 2)
        self.assertEqual(len(scale.shape), 2)

        scale = paddle.repeat_interleave(
            paddle.repeat_interleave(scale, repeats=128, axis=0),
            repeats=128,
            axis=1,
        )
        scale = scale[: x_q.shape[0], : x_q.shape[1]]

        self.assertEqual(scale.shape, x_q.shape)

        x_qdq = x_q.astype('float32') * scale
        rmse = self.cal_all_rmse(x, x_q, x_qdq, input_transpose)

        self.assertLessEqual(rmse, self.rmse_threshold)
        return rmse

    def eval_per_1x128_quant(
        self,
        x: paddle.Tensor,
        input_transpose: bool = False,
        scale_transpose=False,
    ):
        if input_transpose:
            x_q, scale, x_t_q, scale_t = fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=False,
                using_pow2_scale=False,
            )
        else:
            x_q, scale = fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=False,
                using_pow2_scale=False,
            )

        self.assertEqual(len(x_q.shape), 2)
        self.assertEqual(len(scale.shape), 2)

        scale = paddle.repeat_interleave(scale, repeats=128, axis=1)
        scale = scale[: x_q.shape[0], : x_q.shape[1]]

        self.assertEqual(scale.shape, x_q.shape)

        x_qdq = x_q.astype('float32') * scale
        rmse = self.cal_all_rmse(x, x_q, x_qdq, input_transpose)

        self.assertLessEqual(rmse, self.rmse_threshold)
        return rmse

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.bfloat16)

    def test_quantization_consistency_128x128(self):
        paddle.seed(42)
        x1 = paddle.randn((1024, 1024), dtype=paddle.bfloat16)
        rmse1 = self.eval_per_128x128_quant(x1, input_transpose=False)

        paddle.seed(42)
        x2 = paddle.randn((1024, 1024), dtype=paddle.bfloat16)
        rmse2 = self.eval_per_128x128_quant(x2, input_transpose=False)

        self.assertAlmostEqual(rmse1.item(), rmse2.item(), places=6)

    def test_quantization_consistency_1x128(self):
        paddle.seed(42)
        x1 = paddle.randn((1024, 1024), dtype=paddle.bfloat16)
        rmse1 = self.eval_per_1x128_quant(x1, input_transpose=False)

        paddle.seed(42)
        x2 = paddle.randn((1024, 1024), dtype=paddle.bfloat16)
        rmse2 = self.eval_per_1x128_quant(x2, input_transpose=False)

        self.assertAlmostEqual(rmse1.item(), rmse2.item(), places=6)


if __name__ == '__main__':
    unittest.main()
