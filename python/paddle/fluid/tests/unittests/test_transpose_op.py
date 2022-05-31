#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle.fluid.core as core

paddle.enable_static()


class TestTransposeOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.python_api = paddle.transpose
        self.inputs = {'X': np.random.random(self.shape).astype("float64")}
        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {
            'XShape': np.random.random(self.shape).astype("float64"),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = False

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)

    def initTestCase(self):
        self.shape = (3)
        self.axis = (0)


class TestCase0(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 2, 3, 2, 4, 6)
        self.axis = (0, 1, 3, 2, 4, 5, 6)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
