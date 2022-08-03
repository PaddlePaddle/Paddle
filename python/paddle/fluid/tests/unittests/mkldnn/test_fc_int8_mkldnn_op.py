# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool


@OpTestTool.skip_if_not_cpu()
class TestFCINT8OneDNNOp(OpTest):

    def setUp(self):
        self.op_type = "fc"
        self._cpu_only = True
        self.configure()
        self.generate_data()
        self.set_inputs()

        self.attrs = {
            'use_mkldnn': True,
            'Scale_in': self.x_scale,
            'Scale_weights': [self.y_scale],
            'Scale_out': self.out_scale,
            'force_fp32_output': self.force_fp32_output
        }

        if self.force_fp32_output:
            out = self.out_float
        else:
            out = self.out

        self.outputs = {'Out': out}

    def configure(self):
        self.use_bias = True
        self.force_fp32_output = False

    def set_inputs(self):
        self.inputs = {'Input': self.x, 'W': self.y_float, 'Bias': self.bias}

    def quantize(self, tensor):
        scale = 63. / np.abs(np.amax(tensor))
        quantized = np.round(scale * tensor).astype("int8")
        return scale, quantized

    def generate_data(self):
        self.x_float = np.random.random((10, 5)).astype("float32") * 10
        self.x_scale, self.x = self.quantize(self.x_float)

        self.y_float = np.random.random((5, 10)).astype("float32") * 10
        self.y_scale, self.y = self.quantize(self.y_float)

        self.out_float = np.dot(self.x_float, self.y_float)
        if self.use_bias:
            self.bias = np.random.random((10)).astype("float32") * 10
            self.out_float += self.bias

        self.out_scale, self.out = self.quantize(self.out_float)

    def test_check_output(self):
        int_atol = 2
        self.check_output(int_atol)


class TestFCINT8NoBiasOneDNNOp(TestFCINT8OneDNNOp):

    def configure(self):
        self.use_bias = False
        self.force_fp32_output = False

    def set_inputs(self):
        self.inputs = {
            'Input': self.x,
            'W': self.y_float,
        }


class TestFCINT8ForceFP32OutputOneDNNOp(TestFCINT8NoBiasOneDNNOp):

    def configure(self):
        self.use_bias = False
        self.force_fp32_output = True


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    unittest.main()
