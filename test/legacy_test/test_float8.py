# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import re
import unittest

import paddle
from paddle.base import core

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11800,
    "fp8 support in CUDA need CUDA version >= 11.8.",
)
class TestFP8CastOp(unittest.TestCase):
    def setUp(self):
        self.dtype_dict = {
            "float8_e4m3fn": core.VarDesc.VarType.FP8_E4M3FN,
            "float8_e5m2": core.VarDesc.VarType.FP8_E5M2,
        }
        self.shape = (16, 16)

    def test_cast(self):
        for self.device in ["cpu", "gpu"]:
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                # test fp32 to fp8 (dtype)
                input = paddle.full(self.shape, 57345.0)
                input1 = input.astype(self.dtype)
                self.assertTrue(input1.dtype == self.dtype_dict[self.dtype])
                # test fp8 to fp32 (dtype)
                input2 = input1.astype("float32")
                self.assertTrue(input2.dtype == core.VarDesc.VarType.FP32)
                # test fp32 to fp8 (value clip)
                expect = paddle.full(
                    self.shape,
                    E4M3_MAX_POS
                    if self.dtype == "float8_e4m3fn"
                    else E5M2_MAX_POS,
                )
                # there exists some problem in cpu fp8 cast
                if self.device == "gpu":
                    self.assertTrue(paddle.equal_all(input2, expect))


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11800,
    "fp8 support in CUDA need CUDA version >= 11.8.",
)
class TestFP8FullOp(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu")
        self.dtype_dict = {
            "float8_e4m3fn": core.VarDesc.VarType.FP8_E4M3FN,
            "float8_e5m2": core.VarDesc.VarType.FP8_E5M2,
        }

    def test_ones(self):
        for self.device in ["cpu", "gpu"]:
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                input = paddle.ones([1, 2], dtype=self.dtype)
                self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                input_fp32 = input.astype("float32")
                expect = paddle.to_tensor([[1, 1]]).astype("float32")
                # there exists some problem in cpu fp8 full
                if self.device == "gpu":
                    self.assertTrue(paddle.equal_all(expect, input_fp32))

    def test_zeros(self):
        for self.device in ["cpu", "gpu"]:
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                input = paddle.zeros([1, 2], dtype=self.dtype)
                self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                input_fp32 = input.astype("float32")
                expect = paddle.to_tensor([[0, 0]]).astype("float32")
                if self.device == "gpu":
                    self.assertTrue(paddle.equal_all(expect, input_fp32))


if __name__ == "__main__":
    unittest.main()
