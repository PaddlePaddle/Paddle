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
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter

np.random.seed(2013)

import os
import re


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
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "weight_only_linear requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedWeightOnlyLinearPass_Fp32(PassTest):
    def is_program_valid(self, program):
        return True

    def build_ir_progam(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x = paddle.static.data(
                    name='x', shape=[3, 64, 64], dtype=self.dtype
                )
                initializer = paddle.nn.initializer.Constant(0.0)
                w = create_parameter(
                    shape=[64, 64], dtype=self.dtype, initializer=initializer
                )
                bias_ = paddle.static.data(
                    name="bias",
                    shape=[64],
                    dtype=self.dtype,
                )
                bias = paddle.assign(bias_)
                res1 = paddle.matmul(x=x, y=w)
                out = paddle.add(res1, bias)
        self.pass_list = ['fused_weight_only_linear_pass']
        self.feeds = {
            "x": np.random.random((3, 64, 64)).astype(self.dtype),
            "w": np.random.random((64, 64)).astype(self.dtype),
            "bias": np.random.random(64).astype(self.dtype),
        }
        self.fetch_list = [out]

        return pir_program

    def setUp(self):
        self.place_runtime = "gpu"
        self.dtype = 'float32'
        # weight_quantize need weight's dtype to be fp16 or bf16
        self.valid_op_map = {
            "pd_op.weight_only_linear": 0,
            "pd_op.weight_quantize": 0,
            "pd_op.matmul": 1,
            "pd_op.add": 1,
        }

    def sample_program(self):
        yield self.build_ir_progam(), False

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedWeightOnlyLinearPass_Fp16(TestFusedWeightOnlyLinearPass_Fp32):
    def setUp(self):
        self.place_runtime = "gpu"
        self.dtype = 'float16'
        self.valid_op_map = {
            "pd_op.weight_only_linear": 1,
            "pd_op.weight_quantize": 1,
            "pd_op.matmul": 0,
            "pd_op.add": 0,
        }


class TestFusedWeightOnlyLinearPass_wdim_divisible_by_16(
    TestFusedWeightOnlyLinearPass_Fp32
):
    def build_ir_progam(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x = paddle.static.data(
                    name='x', shape=[3, 64, 64], dtype=self.dtype
                )
                initializer = paddle.nn.initializer.Constant(0.0)
                w = create_parameter(
                    shape=[64, 15], dtype=self.dtype, initializer=initializer
                )
                bias_ = paddle.static.data(
                    name="bias",
                    shape=[15],
                    dtype=self.dtype,
                )
                bias = paddle.assign(bias_)
                res1 = paddle.matmul(x=x, y=w)
                out = paddle.add(res1, bias)
        self.pass_list = ['fused_weight_only_linear_pass']
        self.feeds = {
            "x": np.random.random((3, 64, 64)).astype(self.dtype),
            "w": np.random.random((64, 15)).astype(self.dtype),
            "bias": np.random.random(15).astype(self.dtype),
        }
        self.fetch_list = [out]

        return pir_program

    def setUp(self):
        self.place_runtime = "gpu"
        self.dtype = 'float16'
        self.valid_op_map = {
            "pd_op.weight_only_linear": 0,
            "pd_op.weight_quantize": 0,
            "pd_op.matmul": 1,
            "pd_op.add": 1,
        }


if __name__ == "__main__":
    unittest.main()
