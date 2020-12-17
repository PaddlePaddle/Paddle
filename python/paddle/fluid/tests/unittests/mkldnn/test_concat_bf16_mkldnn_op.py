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
class TestConcatBf16Op(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.init_axis()
        self.init_shape()
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {
            'axis': self.axis,
            'use_mkldnn': True,
            'mkldnn_data_type': self.mkldnn_data_type
        }

        self.output = np.concatenate(
            (self.x0, self.x1, self.x2), axis=self.axis).astype(np.uint16)
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

# --------------------test concat bf16 in with axis 0--------------------

    def init_test_data(self):
        self.x0 = convert_float_to_uint16(
            np.random.random(self.x0_shape).astype(np.float32))
        self.x1 = convert_float_to_uint16(
            np.random.random(self.x1_shape).astype(np.float32))
        self.x2 = convert_float_to_uint16(
            np.random.random(self.x2_shape).astype(np.float32))

    def init_axis(self):
        self.axis = 0

    def init_shape(self):
        self.x0_shape = [2, 2, 1, 2]
        self.x1_shape = [1, 2, 1, 2]
        self.x2_shape = [3, 2, 1, 2]


# --------------------test concat bf16 in with axis 1--------------------


class TestAxis1Case(TestConcatBf16Op):
    def init_axis(self):
        self.axis = 1

    def init_shape(self):
        self.x0_shape = [1, 1, 5, 5]
        self.x1_shape = [1, 2, 5, 5]
        self.x2_shape = [1, 3, 5, 5]


# --------------------test concat bf16 in with axis 2--------------------


class TestAxis2Case(TestConcatBf16Op):
    def init_axis(self):
        self.axis = 2

    def init_shape(self):
        self.x0_shape = [2, 3, 4, 5]
        self.x1_shape = [2, 3, 5, 5]
        self.x2_shape = [2, 3, 6, 5]


# --------------------test concat bf16 in with axis 3--------------------


class TestAxis3Case(TestConcatBf16Op):
    def init_axis(self):
        self.axis = 3

    def init_shape(self):
        self.x0_shape = [2, 3, 5, 5]
        self.x1_shape = [2, 3, 5, 6]
        self.x2_shape = [2, 3, 5, 7]


if __name__ == '__main__':
    enable_static()
    unittest.main()
