#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.base import core

is_sm_supported = (
    paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] == 9
) or (paddle.device.cuda.get_device_capability()[0] >= 9)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11080
    or not is_sm_supported,
    "MatmulFp8 requires CUDA >= 11.8 and SM version >= 8.9",
)
class TestMatmulFp8(unittest.TestCase):
    def config(self):
        self.dtype = 'float8_e4m3fn'
        self.rtol = 0.7
        self.atol = 2.5
        self.x_shape = (32, 16)
        self.y_shape = (16, 32)

    def setUp(self):
        self.config()
        np.random.seed(2014)
        self.input_a_np = np.random.random(self.x_shape)
        self.input_b_np = np.random.random(self.y_shape)
        self.input_a = paddle.to_tensor(self.input_a_np).astype(self.dtype)
        self.input_b = paddle.to_tensor(self.input_b_np).astype(self.dtype)

    def get_reference_out(self):
        out = np.matmul(self.input_a_np, self.input_b_np)
        return out

    def get_op_out(self):
        out = paddle.matmul(self.input_a, self.input_b)
        return out.numpy()

    def test_matmul_fp8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()
