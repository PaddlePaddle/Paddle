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
import re
import unittest

import paddle
from paddle.base import core


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
class Float8_E4M3FN_Test_GPU(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu")
        self.dtype = "float8_e4m3fn"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E4M3FN

    def test_fullOp(self):
        input1 = paddle.ones([16, 16], dtype=self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)

    def test_castOp(self):
        input1 = paddle.ones([16, 16])
        input1 = input1.astype(self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)

        inut2 = input1.astype("float32")
        self.assertTrue(inut2.dtype == core.VarDesc.VarType.FP32)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11800,
    "fp8 support in CUDA need CUDA version >= 11.8.",
)
class Float8_E5M2_Test_GPU(Float8_E4M3FN_Test_GPU):
    def setUp(self):
        paddle.device.set_device("gpu")
        self.dtype = "float8_e5m2"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E5M2


class Float8_E4M3FN_Test_CPU(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("cpu")
        self.dtype = "float8_e4m3fn"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E4M3FN

    def test_fullOp(self):
        input1 = paddle.ones([16, 16], dtype=self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)

    def test_castOp(self):
        input1 = paddle.ones([16, 16])
        input1 = input1.astype(self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)

        inut2 = input1.astype("float32")
        self.assertTrue(inut2.dtype == core.VarDesc.VarType.FP32)


class Float8_E5M2_Test_CPU(Float8_E4M3FN_Test_CPU):
    def setUp(self):
        paddle.device.set_device("cpu")
        self.dtype = "float8_e5m2"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E5M2


if __name__ == "__main__":
    unittest.main()
