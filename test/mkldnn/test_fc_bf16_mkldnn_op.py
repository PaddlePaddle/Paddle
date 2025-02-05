# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

from paddle import enable_static
from paddle.base import core


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights) + bias_data
    return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype(np.float32)
        self.weights = np.random.random((ic * h * w, oc)).astype(np.float32)


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestFcBf16MklDNNOp(OpTest):
    def generate_data(self):
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype("float32")

    def setUp(self):
        self.op_type = "fc"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.generate_data()

        self.output = fully_connected_naive(
            self.matrix.input, self.matrix.weights, self.bias
        )
        if not self.force_fp32_output:
            self.output = convert_float_to_uint16(self.output)

        self.inputs = {
            'Input': convert_float_to_uint16(self.matrix.input),
            'W': self.matrix.weights,
            'Bias': self.bias,
        }

        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
            'force_fp32_output': self.force_fp32_output,
        }

        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCMKLDNNOp1(TestFcBf16MklDNNOp):
    def generate_data(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype(np.float32)


if __name__ == "__main__":
    enable_static()
    unittest.main()
