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


def get_output(X, Y, Bias, Act):
    out = np.dot(X, Y) + Bias
    return out


@skip_check_grad_ci(reason="no grad op")
class TestCutlassInt4GemmOp(OpTest):
    def setUp(self):
        self.op_type = "int4_gemm_cutlass"
        self.python_api = paddle.incubate.nn.functional.int4_gemm_cutlass
        self.place = core.CUDAPlace(0)
        self.dtype = np.int8
        self.x = np.ones((128, 64), dtype=self.dtype)
        self.y = np.ones((64, 256), dtype=self.dtype)
        self.bias = np.ones((256), dtype=self.dtype)
        self.inputs = {'X': self.x, 'Y': self.y, 'Bias': self.bias}
        self.attrs = {"activation": "none"}
        self.outputs = {'Out': get_output(self.x, self.y, self.bias, 'none')}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(42)
    unittest.main()
