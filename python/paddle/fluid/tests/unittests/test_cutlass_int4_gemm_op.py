# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid.core as core
from paddle.incubate.nn.functional import fused_matmul_bias_int4


def get_output(X, Y, Bias, Act):
    if Act == 'none':
        out = np.dot(X, Y) + Bias
    elif Act == 'relu':
        out = np.dot(X, Y) + Bias
        out = np.maximum(out, 0)
    return out


@skip_check_grad_ci(reason="no grad op")
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7,
    "CutlassInt4Gemm requires CUDA_ARCH>=75",
)
class TestCutlassInt4GemmOp(OpTest):
    def setUp(self):
        self.op_type = "int4_gemm_cutlass"
        self.__class__.op_type = "int4_gemm_cutlass"
        self.place = paddle.CUDAPlace(0)
        self.dtype = np.int32
        self.config()
        self.x = np.random.randint(-7, 7, size=(128, 64)).astype(self.dtype)
        self.y = np.random.randint(-7, 7, size=(64, 128)).astype(self.dtype)
        self.bias = np.random.randint(-7, 7, size=(128)).astype(self.dtype)
        self.inputs = {
            'X': paddle.to_tensor(self.x, place=self.place),
            'Y': paddle.to_tensor(self.y, place=self.place),
            'Bias': paddle.to_tensor(self.bias, place=self.place),
        }
        self.outputs = {
            'Out': get_output(
                self.x, self.y, self.bias, self.attrs['activation']
            )
        }

    def config(self):
        self.attrs = {"trans_x": False, "trans_y": False, "activation": "none"}

    def get_int4_gemm_out(self):
        gemm_out = fused_matmul_bias_int4(
            self.inputs['X'],
            self.inputs['Y'],
            self.inputs['Bias'],
            self.attrs['trans_x'],
            self.attrs['trans_y'],
            self.attrs['activation'],
        )
        return gemm_out

    def test_check_output(self):
        out_ref = self.outputs['Out']
        out = self.get_int4_gemm_out().numpy()
        np.testing.assert_allclose(out_ref, out, rtol=1e-3, atol=1e-3)


@skip_check_grad_ci(reason="no grad op")
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7,
    "CutlassInt4Gemm requires CUDA_ARCH>=75",
)
class TestCutlassInt4GemmOpRelu(TestCutlassInt4GemmOp):
    def config(self):
        super().config()
        self.attrs['activation'] = 'relu'


if __name__ == "__main__":
    np.random.seed(42)
    unittest.main()
