# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_fill_constant_op import TestFillConstantOp1, TestFillConstantOp2, TestFillConstantOpWithSelectedRows


class TestNGRAPHFillConstantOp1(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.dtype = np.float64

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8, 'dtype': 6}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def test_check_output(self):
        self.check_output()


class TestNGRAPHFillConstantOp2(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.dtype = np.int32

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 2}
        self.outputs = {'Out': np.full((123, 92), 0)}

    def test_check_output(self):
        self.check_output()


class TestNGRAPHFillConstantOp3(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.dtype = np.int64

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'dtype': 3}
        self.outputs = {'Out': np.full((123, 92), 0)}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
