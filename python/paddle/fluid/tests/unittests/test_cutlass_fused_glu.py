# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import OpTest

# Ensure we use float type to accumulate
os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "0"

import paddle
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import cutlass_fused_glu

default_main_program().random_seed = 42


class TestCutlassFusedGluOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-3
        self.atol = 1e-4

        self.paddle_dtype = paddle.float32
        if self.x_type == np.float16:
            self.paddle_dtype = paddle.float16

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "cutlass_fused_glu"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        self.x = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[self.batch, self.in_channels],
            ),
            dtype=self.paddle_dtype,
        )
        self.weight = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[self.in_channels, self.in_channels * 2],
            ),
            dtype=self.paddle_dtype,
        )
        self.bias = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[self.in_channels * 2],
            ),
            dtype=self.paddle_dtype,
        )

        self.x.stop_gradient = True
        self.weight.stop_gradient = True
        self.bias.stop_gradient = True

    def config(self):
        self.x_type = np.float16
        self.batch = 64
        self.in_channels = 256
        self.act_type = "swish"

    def GetBaselineOut(self):
        paddle.disable_static()

        out = paddle.matmul(self.x, self.weight)
        out = out + self.bias
        x0, x1 = paddle.chunk(out, 2, axis=1)
        if self.act_type == "swish":
            x1 = paddle.nn.functional.swish(x1)
        elif self.act_type == "sigmoid":
            x1 = paddle.nn.functional.sigmoid(x1)
        elif self.act_type == "gelu":
            x1 = paddle.nn.functional.gelu(x1)
        else:
            x1 = x1
        return x0 * x1

    def GetFusedGLUOut(self):
        paddle.disable_static()
        fused_out = cutlass_fused_glu(
            self.x, self.weight, self.bias, self.act_type
        )

        return fused_out

    def test_cutlass_fused_glu_op(self):
        print("==== here is test ====")
        final_out_ref = self.GetBaselineOut()
        final_out = self.GetFusedGLUOut()

        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


class TestCutlassGLUSigmoidOpFp16(TestCutlassFusedGluOp):
    def config(self):
        super().config()
        self.batch = 64
        self.in_channels = 128
        self.act_type = "sigmoid"


class TestCutlassGLUSwishOpFp16(TestCutlassFusedGluOp):
    def config(self):
        super().config()
        self.batch = 16
        self.in_channels = 64
        self.act_type = "swish"


class TestCutlassGLUGeluOpFp16(TestCutlassFusedGluOp):
    def config(self):
        super().config()
        self.batch = 32
        self.in_channels = 256
        self.act_type = "gelu"


if __name__ == "__main__":
    unittest.main()
