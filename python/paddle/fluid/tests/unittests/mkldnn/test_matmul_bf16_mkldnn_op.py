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

from __future__ import print_function

import unittest
import os
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
from paddle import enable_static


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestMatmulBf16MklDNNOp(OpTest):
    def generate_data(self):
        self.x = np.random.random((25, 2, 2)).astype(np.float32)
        self.y = np.random.random((25, 2, 2)).astype(np.float32)
        self.alpha = 1.0
        self.out = self.alpha * np.matmul(self.x, self.y)

    def set_attributes(self):
        self.alpha = self.alpha if hasattr(self, 'alpha') else 1.0
        self.attrs = {
            'alpha': self.alpha,
            "use_mkldnn": self.use_mkldnn,
            "mkldnn_data_type": self.mkldnn_data_type,
            "force_fp32_output": self.force_fp32_output
        }

    def setUp(self):
        self.op_type = "matmul"
        self.use_mkldnn = True
        self.dtype = np.uint16
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.generate_data()
        self.set_attributes()

        if not self.force_fp32_output:
            self.out = convert_float_to_uint16(self.out)
        self.outputs = {'Out': self.out}

        self.x = convert_float_to_uint16(self.x)
        self.y = convert_float_to_uint16(self.y)
        self.inputs = {'X': self.x, 'Y': self.y}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass


class TestDnnlMatMulOpAlpha(TestMatmulBf16MklDNNOp):
    def generate_data(self):
        self.x = np.random.random((17, 2, 3)).astype(np.float32)
        self.y = np.random.random((17, 3, 2)).astype(np.float32)
        self.alpha = 2.0
        self.out = self.alpha * np.matmul(self.x, self.y)


class TestDnnlMatMulOp2D(TestMatmulBf16MklDNNOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype(np.float32)
        self.y = np.random.random((9, 12)).astype(np.float32)
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpTransposeX(TestMatmulBf16MklDNNOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype(np.float32)
        self.y = np.random.random((12, 9)).astype(np.float32)
        self.out = np.matmul(np.transpose(self.x), self.y)

    def set_attributes(self):
        self.attrs = {
            "use_mkldnn": self.use_mkldnn,
            "mkldnn_data_type": self.mkldnn_data_type,
            'transpose_X': True
        }


class TestDnnlMatMulOpTransposeY(TestMatmulBf16MklDNNOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype(np.float32)
        self.y = np.random.random((12, 9)).astype(np.float32)
        self.out = np.matmul(self.x, np.transpose(self.y))

    def set_attributes(self):
        self.attrs = {
            "use_mkldnn": self.use_mkldnn,
            "mkldnn_data_type": self.mkldnn_data_type,
            'transpose_Y': True
        }


class TestMatmulBf16MklDNNForceFp32Output(TestMatmulBf16MklDNNOp):
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype(np.float32)
        self.y = np.random.random((9, 12)).astype(np.float32)
        self.force_fp32_output = True
        self.alpha = 0.5
        self.out = self.alpha * np.matmul(self.x, self.y)


if __name__ == "__main__":
    enable_static()
    unittest.main()
