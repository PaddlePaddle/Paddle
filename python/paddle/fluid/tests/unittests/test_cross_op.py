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
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestCrossDefaultOp(OpTest):
    def setUp(self):
        self.op_type = "cross"
        self.dtype = np.float64
        self.inputs = {
            'X': np.random.random((100, 3, 1)).astype(self.dtype),
            'Y': np.random.random((100, 3, 1)).astype(self.dtype)
        }
        self.init_output()

    def init_output(self):
        x = np.squeeze(self.inputs['X'], 2)
        y = np.squeeze(self.inputs['Y'], 2)
        z_list = []
        for i in range(100):
            z_list.append(np.cross(x[i], y[i]))
        self.outputs = {'Out': np.array(z_list).reshape(100, 3, 1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestCrossOp(OpTest):
    def setUp(self):
        self.op_type = "cross"
        self.dtype = np.float64
        self.inputs = {
            'X': np.random.random((100, 3, 1)).astype(self.dtype),
            'Y': np.random.random((100, 3, 1)).astype(self.dtype)
        }
        self.attrs = {'dim': -2}
        self.init_output()

    def init_output(self):
        x = np.squeeze(self.inputs['X'], 2)
        y = np.squeeze(self.inputs['Y'], 2)
        z_list = []
        for i in range(100):
            z_list.append(np.cross(x[i], y[i]))
        self.outputs = {'Out': np.array(z_list).reshape(100, 3, 1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == '__main__':
    unittest.main()
