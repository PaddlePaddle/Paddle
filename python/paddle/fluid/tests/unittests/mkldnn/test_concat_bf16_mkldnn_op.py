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

from __future__ import print_function

import unittest
import numpy as np
import struct

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
from paddle import enable_static


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestConcatOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.mkldnn_data_type = "float32"
        self.init_axis()
        self.init_shape()
        self.init_test_data()
        self.configure_datatype()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {
            'axis': self.axis,
            'use_mkldnn': True,
            'mkldnn_data_type': self.mkldnn_data_type
        }

        self.output = np.concatenate(
            (self.x0, self.x1, self.x2), axis=self.axis).astype(np.float32)
        self.outputs = {'Out': self.output}

    def configure_datatype(self):
        self.mkldnn_data_type = "float32"

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')
        self.check_grad(['x1'], 'Out')
        self.check_grad(['x2'], 'Out')

# --------------------test concat bf16 in with axis 0--------------------

    def init_test_data(self):
        self.x0 = np.random.random(self.x0_shape).astype(np.float32)
        self.x1 = np.random.random(self.x1_shape).astype(np.float32)
        self.x2 = np.random.random(self.x2_shape).astype(np.float32)

    def init_axis(self):
        self.axis = 0

    def init_shape(self):
        self.x0_shape = [2, 2, 1, 30]
        self.x1_shape = [1, 2, 1, 30]
        self.x2_shape = [3, 2, 1, 30]


# --------------------test concat bf16 in with axis 1--------------------


class TestAxis1Case(TestConcatOneDNNOp):
    def init_axis(self):
        self.axis = 1

    def init_shape(self):
        self.x0_shape = [1, 1, 5, 50]
        self.x1_shape = [1, 2, 5, 50]
        self.x2_shape = [1, 3, 5, 50]


# --------------------test concat bf16 in with axis 2--------------------


class TestAxis2Case(TestConcatOneDNNOp):
    def init_axis(self):
        self.axis = 2

    def init_shape(self):
        self.x0_shape = [2, 3, 4, 50]
        self.x1_shape = [2, 3, 5, 50]
        self.x2_shape = [2, 3, 6, 50]


# --------------------test concat bf16 in with axis 3--------------------


class TestAxis3Case(TestConcatOneDNNOp):
    def init_axis(self):
        self.axis = 3

    def init_shape(self):
        self.x0_shape = [5, 3, 5, 5]
        self.x1_shape = [5, 3, 5, 6]
        self.x2_shape = [5, 3, 5, 7]


#   BF16 TESTS
def create_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestConcatBF16OneDNNOp(parent):
        def set_inputs(self):
            self.x_fp32 = np.random.random((10, 10)).astype(np.float32) * 25
            self.inputs = {'X': convert_float_to_uint16(self.x_fp32)}

        def configure_datatype(self):
            self.mkldnn_data_type = "bfloat16"
            self.x0 = convert_float_to_uint16(self.x0)
            self.x1 = convert_float_to_uint16(self.x1)
            self.x2 = convert_float_to_uint16(self.x2)

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dxs = np.split(self.dout, axis=self.axis)

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["x0"],
                "Out",
                user_defined_grads=[self.dxs[0]],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    cls_name = "{0}_{1}".format(parent.__name__, "BF16")
    TestClipBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestClipBF16OneDNNOp

if __name__ == '__main__':
    enable_static()
    unittest.main()
