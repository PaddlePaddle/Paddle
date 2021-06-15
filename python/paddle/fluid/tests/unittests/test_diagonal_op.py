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
from op_test import OpTest
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.tensor as tensor

paddle.enable_static()


class TestDiagonalOp(OpTest):
    def setUp(self):
        self.op_type = "diagonal"
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.case = np.random.randn(20, 6, 3, 5, 4).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])


class TestDiagonalOpCase1(TestDiagonalOp):
    def init_config(self):
        self.case = np.random.randn(8, 20, 6, 4).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 3, 'axis2': 0}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])


class TestDiagonalOpCase2(TestDiagonalOp):
    def init_config(self):
        self.case = np.random.randn(6, 20, 7, 4).astype('int32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -3, 'axis1': 0, 'axis2': -1}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])


if __name__ == '__main__':
    unittest.main()
