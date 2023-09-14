#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "MatmulInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestMatmulInt8(unittest.TestCase):
    """
    Test matmul int8
    Only NT (Non-Transposed-A and Transposed-B) is supported
    """

    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.m = 8
        self.k = 64
        self.n = 64

    def setUp(self):
        self.config()
        self.input_a_np = np.random.randint(-127, 127, [self.m, self.k]).astype(
            'int32'
        )
        self.input_b_np = np.random.randint(-127, 127, [self.k, self.n]).astype(
            'int32'
        )
        self.input_a = paddle.to_tensor(self.input_a_np, dtype=self.dtype)
        self.input_b = paddle.to_tensor(
            self.input_b_np.transpose((1, 0)), dtype=self.dtype
        )

    def get_reference_out(self):
        out = np.dot(self.input_a_np, self.input_b_np)
        return out

    def get_op_out(self):
        out = paddle._C_ops.matmul_int8(self.input_a, self.input_b, False, True)
        return out.numpy()

    def test_matmul_int8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()
