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

import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from mkldnn_op_test import format_reorder


class TestTransposeOp(OpTest):

    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.initInputData()
        self.use_mkldnn = True
        self._cpu_only = True
        self.axis = (0, 2, 3, 1)

        self.inputs = {
            'X': format_reorder(self.input_data, self.shape).astype(np.int8)
        }  #transform data format to 'NHWC' for INT8 transpose specially.

        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
        }

        self.outputs = {
            'XShape': np.random.random(self.shape).astype(np.int8),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def init_op_type(self):
        self.op_type = "transpose2"

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output_with_place(core.CPUPlace(),
                                     1e-5,
                                     no_check_set=['XShape'],
                                     check_dygraph=False)

    def initTestCase(self):
        self.shape = (2, 3, 4, 5)

    def initInputData(self):
        self.input_data = (np.random.randint(0, 100, self.shape) - 50).astype(
            np.int8)


class TestINT8Case(TestTransposeOp):

    def initTestCase(self):
        self.shape = (2, 4, 6, 8)

    def initInputData(self):
        self.input_data = (np.random.randint(0, 100, self.shape) - 50).astype(
            np.int8)


class TestUINT8Case(TestTransposeOp):

    def initTestCase(self):
        self.shape = (1, 3, 5, 7)

    def initDataType(self):
        self.input_data = (np.random.randint(0, 100,
                                             self.shape)).astype(np.uint8)


if __name__ == '__main__':
    unittest.main()
