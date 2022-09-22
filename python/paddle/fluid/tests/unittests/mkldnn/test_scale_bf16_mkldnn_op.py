#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
@unittest.skipIf(core.is_compiled_with_cuda(),
                 "core is compiled with CUDA which has no BF implementation")
class TestScaleOpBF16(OpTest):

    def setUp(self):
        self.op_type = "scale"
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale = -2.3
        self.inputs = {'X': self.x_bf16}
        self.attrs = {'scale': self.scale, 'use_mkldnn': True, 'bias': 0.4}
        self.use_mkldnn = True
        self.outputs = {
            'Out': (self.x_fp32 * self.attrs['scale']) + self.attrs['bias']
        }

    def calculate_grads(self):
        bias = 0
        if 'bias' in self.attrs:
            bias = self.attrs['bias']

        scale = self.scale
        if 'ScaleTensor' in self.attrs:
            scale = self.attrs['ScaleTensor']

        self.out = (self.x_fp32 * scale) + bias
        self.dx = (self.out * scale)

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ["X"],
            "Out",
            check_dygraph=False,
            user_defined_grads=[self.dx],
            user_defined_grad_outputs=[convert_float_to_uint16(self.out)])


class TestScaleOpBF16BiasNotAfterScale(TestScaleOpBF16):

    def setUp(self):
        self.op_type = "scale"
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale = 1.5
        self.inputs = {'X': self.x_bf16}
        self.attrs = {
            'scale': self.scale,
            'use_mkldnn': True,
            'bias': 0.0,
            'bias_after_scale': False
        }
        self.use_mkldnn = True
        self.outputs = {
            'Out': (self.x_fp32 + self.attrs['bias']) * self.attrs['scale']
        }


class TestScaleOpBF16ScaleTensor(TestScaleOpBF16):

    def setUp(self):
        self.op_type = "scale"
        self.scale = -2.3
        self.x_fp32 = np.random.random((10, 10)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale_tensor = np.array([self.scale]).astype(np.float32)
        self.inputs = {
            'X': self.x_bf16,
            'ScaleTensor': convert_float_to_uint16(self.scale_tensor)
        }
        self.attrs = {'use_mkldnn': True}
        self.outputs = {'Out': self.x_fp32 * self.scale}


class TestScaleOpBF16ScaleTensorNotBiasAfterScale(TestScaleOpBF16):

    def setUp(self):
        self.op_type = "scale"
        self.scale = 1.2
        self.x_fp32 = np.random.random((9, 13)).astype(np.float32)
        self.x_bf16 = convert_float_to_uint16(self.x_fp32)
        self.scale_tensor = np.array([self.scale]).astype(np.float32)
        self.inputs = {
            'X': self.x_bf16,
            'ScaleTensor': convert_float_to_uint16(self.scale_tensor)
        }
        self.attrs = {
            'bias': -1.1,
            'bias_after_scale': False,
            'use_mkldnn': True
        }
        self.outputs = {'Out': (self.x_fp32 + self.attrs['bias']) * self.scale}


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
