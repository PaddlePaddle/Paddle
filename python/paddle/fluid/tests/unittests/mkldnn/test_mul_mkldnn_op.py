# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
import numpy as np
from paddle.fluid.tests.unittests.test_mul_op import TestMulOp, TestMulOp2
'''
 test case for s8 * s8
'''


class TestMKLDNNMulOpS8S8(TestMulOp):
    def setUp(self):
        self.op_type = "mul"
        self.init_kernel_type()
        self.init_data()
        self.attrs = {
            "use_mkldnn": self.use_mkldnn,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_out": self.scale_out,
            "force_fp32_output": self.force_fp32,
        }

    def init_kernel_type(self):
        self.use_mkldnn = True
        self.force_fp32 = True
        self.dsttype = np.float32 if self.force_fp32 else np.int8

    def init_data(self):
        self.scale_x = 1.0
        self.scale_y = [1.0]
        self.scale_out = 1.0

        A_data_s8 = np.array([[-47, -123, -10, 92], [-96, -38, 82, 40],
                              [70, -60, -66, -127],
                              [-46, -91, 109, 101]]).astype(np.int8)

        B_data_s8 = np.array([[124, 8, -74, 32], [-40, 4, -50, -84],
                              [103, -84, 88, -66],
                              [24, 115, 123, 54]]).astype(np.float32)

        quant_B = np.round(B_data_s8 * self.scale_y[0]).astype(np.int)
        output = np.dot(A_data_s8, quant_B)

        scale_output_shift = (self.scale_out) / \
            (self.scale_x * self.scale_y[0])
        output = np.round(output * scale_output_shift).astype(self.dsttype)

        self.inputs = {'X': A_data_s8, 'Y': B_data_s8}

        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), atol=0)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


'''
 test case for  s8 * u8
'''


class TestMKLDNNMulOpS8U8(TestMKLDNNMulOpS8S8):
    def setUp(self):
        self.op_type = "mul"
        self.init_kernel_type()
        self.init_data()
        self.attrs = {
            "use_mkldnn": self.use_mkldnn,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_out": self.scale_out,
            "force_fp32_output": self.force_fp32,
        }

    def init_data(self):
        self.scale_x = 1.0
        self.scale_y = [1.0]
        self.scale_out = 1.0

        # 252 to 152 to avoid overflow clip
        A_data_u8 = np.array([[152, 136, 54, 160], [88, 132, 78, 44],
                              [231, 44, 216, 62],
                              [52, 243, 251, 182]]).astype(np.uint8)

        B_data_s8 = np.array([[-47, -123, -10, 92], [-96, -38, 82, 40],
                              [70, -60, -66, -127],
                              [-46, -91, 109, 101]]).astype(np.float32)

        quant_B = np.round(B_data_s8 * self.scale_y[0]).astype(np.int)
        output = np.dot(A_data_u8, quant_B)

        scale_output_shift = (self.scale_out) / \
            (self.scale_x * self.scale_y[0])
        output = np.round(output * scale_output_shift).astype(self.dsttype)

        self.inputs = {'X': A_data_u8, 'Y': B_data_s8}

        self.outputs = {'Out': output}


'''
 test case for  s8 * u7
'''


class TestMKLDNNMulOpS8U7(TestMKLDNNMulOpS8S8):
    def setUp(self):
        self.op_type = "mul"
        self.init_kernel_type()
        self.init_data()
        self.attrs = {
            "use_mkldnn": self.use_mkldnn,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_out": self.scale_out,
            "force_fp32_output": self.force_fp32,
        }

    def init_data(self):
        self.scale_x = 1.0
        self.scale_y = [1.0]
        self.scale_out = 1.0

        A_data_u7 = np.array([[127, 125, 54, 126], [88, 122, 78, 44],
                              [125, 44, 126, 62],
                              [126, 123, 121, 120]]).astype(np.uint8)

        B_data_s8 = np.array([[-47, -123, -10, 92], [-96, -38, 82, 40],
                              [70, -60, -66, -127],
                              [-46, -91, 109, 101]]).astype(np.float32)

        quant_B = np.round(B_data_s8 * self.scale_y[0]).astype(np.int)
        output = np.dot(A_data_u7, quant_B)

        scale_output_shift = (self.scale_out) / \
            (self.scale_x * self.scale_y[0])
        output = np.round(output * scale_output_shift).astype(self.dsttype)

        self.inputs = {'X': A_data_u7, 'Y': B_data_s8}

        self.outputs = {'Out': output}


'''
 test case for fp32 * fp32
'''


class TestMKLDNNMulOpFP32(TestMKLDNNMulOpS8S8):
    def setUp(self):
        self.op_type = "mul"
        self.use_mkldnn = True

        input_data = np.random.uniform(-100, 100, (3, 3)).astype(np.float32)
        weight = np.random.uniform(-100, 100, (3, 3)).astype(np.float32)
        output = np.dot(input_data, weight)

        self.inputs = {'X': input_data, 'Y': weight}

        self.outputs = {'Out': output}

        self.attrs = {"use_mkldnn": self.use_mkldnn, }

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), atol=0)


if __name__ == '__main__':
    unittest.main()
