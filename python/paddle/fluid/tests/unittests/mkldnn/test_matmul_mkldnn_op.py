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
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest


class TestDnnlMatMulOp(OpTest):
    def generate_data(self):
        self.x = np.random.random((1, 2, 2)).astype("float32")
        self.y = np.random.random((1, 2, 2)).astype("float32")
        self.alpha = 1.0
        self.out = self.alpha * np.matmul(self.x, self.y)

    def set_attributes(self):
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'alpha': self.alpha}

    def setUp(self):
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


# class TestDnnlMatMulOpAlpha(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.random((1, 2, 3)).astype("float32")
#         self.y = np.random.random((1, 3, 2)).astype("float32")
#         self.alpha = 2.0
#         self.out = self.alpha * np.matmul(self.x, self.y)

# class TestDnnlMatMulOp2D(TestDnnlMatMulOp):
#     def print_tensor(self, name, tensor):
#         print(name)
#         print(tensor)

#     def generate_data(self):
#         self.x = np.random.random((3, 2)).astype("float32")
#         self.print_tensor("x", self.x)
#         self.y = np.random.random((2, 3)).astype("float32")
#         self.print_tensor("y", self.y)
#         self.out = np.matmul(self.x, self.y)
#         self.print_tensor("out", self.out)

#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn}

# class TestDnnlMatMulOpTransposeX(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.random((3, 2)).astype("float32")
#         self.y = np.random.random((3, 2)).astype("float32")
#         self.out = np.matmul(np.transpose(self.x), self.y)

#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn, 'transpose_X': True}

# class TestDnnlMatMulOpTransposeY(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.random((3, 2)).astype("float32")
#         self.y = np.random.random((3, 2)).astype("float32")
#         self.out = np.matmul(self.x, np.transpose(self.y))

#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn, 'transpose_Y': True}

# class TestDnnlMatMulOpTransposeY3D(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.random((1, 3, 2)).astype("float32")
#         self.y = np.random.random((1, 3, 2)).astype("float32")
#         self.out = np.matmul(self.x, np.transpose(self.y, (0, 2, 1)))

#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn, 'transpose_Y': True}

# class TestDnnlMatMulOpInt8NoScales(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.random((3, 2)).astype("int8")
#         self.y = np.random.random((2, 3)).astype("int8")
#         self.out = np.matmul(self.x, self.y)
#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn}


class TestDnnlMatMulOpInt8(TestDnnlMatMulOp):
    def print_tensor_info(self, title, floatx, intx, scale):
        print(title)
        print("Float: ")
        print(floatx)
        print("int: ")
        print(intx)
        print("scale: ")
        print(scale)

    def quantize(self, tensor):
        scale = 127. / np.abs(np.amax(tensor))
        quantized = (scale * tensor).astype("int8")
        return scale, quantized

    def generate_data(self):
        x_float = np.array([[1., 1.], [0., 1.]]).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)
        self.print_tensor_info("X", x_float, self.x, self.x_scale)

        y_float = np.array([[1., 1.], [0., 1.]]).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)
        self.print_tensor_info("Y", y_float, self.y, self.y_scale)

        out_float = np.matmul(x_float, y_float)
        self.out_scale, self.out = self.quantize(out_float)
        self.print_tensor_info("out", out_float, self.out, self.out_scale)

    def set_attributes(self):
        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'Scale_out': self.out_scale,
            'force_fp32_output': True
            # 'transpose_Y': True
        }


# class TestDnnlMatMulOpInt8ForceFP32(TestDnnlMatMulOpInt8):

#     def generate_data(self):
#         x_float = np.array([[500., 400.], [300., 350.]]).astype("float32")
#         self.x_scale, self.x = self.quantize(x_float)
#         self.print_tensor_info("X", x_float, self.x, self.x_scale)
#         y_float = np.array([[450., 250.], [550., 300.]]).astype("float32")
#         self.y_scale, self.y = self.quantize(y_float)
#         self.print_tensor_info("Y", y_float, self.y, self.y_scale)
#         out_float = np.matmul(x_float, y_float)
#         self.out = out_float

#     def set_attributes(self):
#         self.attrs = {
#             'use_mkldnn': self.use_mkldnn,
#             'Scale_x': self.x_scale,
#             'Scale_y': self.y_scale,
#             'force_fp32_output' : True
#             }

# class TestDnnlMatMulOpInt8ForceFP32BasicScales(TestDnnlMatMulOp):
#     def generate_data(self):
#         self.x = np.random.randint(0, 10, (3, 2)).astype("int8")
#         self.y = np.random.randint(0, 10, (2, 3)).astype("int8")
#         self.out = np.matmul(self.x, self.y).astype("float32")

#     def set_attributes(self):
#         self.attrs = {'use_mkldnn': self.use_mkldnn, 'force_fp32_output' : True}

if __name__ == "__main__":
    unittest.main()
