#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid

op_function_map = {
    'equal': np.equal,
    'not_equal': np.not_equal,
    'greater_than': np.greater,
    'greater_equal': np.greater_equal,
    'less_than': np.less,
    'less_equal': np.less_equal
}

class TestEqualOneDNNOp(OpTest):
    def set_op_type(self):
        self.op_type = "equal"

    def setUp(self):
        self.set_op_type()
        self.x = np.random.random((10, 10)).astype(np.float32)
        self.y = np.random.random((10, 10)).astype(np.float32)
        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs = {'use_mkldnn': True}
        self.outputs = {'Out': op_function_map[self.op_type](self.x, self.y)}

    def test_check_output(self):
        self.check_output_with_place(fluid.core.CPUPlace())

class TestNotEqualOneDNNOp(TestEqualOneDNNOp):
    def set_op_type(self):
        self.op_type = "not_equal"

class TestGreaterThanOneDNNOp(TestEqualOneDNNOp):
    def set_op_type(self):
        self.op_type = "greater_than"

class TestGreaterEqualOneDNNOp(TestEqualOneDNNOp):
    def set_op_type(self):
        self.op_type = "greater_equal"

class TestLessThanOneDNNOp(TestEqualOneDNNOp):
    def set_op_type(self):
        self.op_type = "less_than"

class TestLessEqualOneDNNOp(TestEqualOneDNNOp):
    def set_op_type(self):
        self.op_type = "less_equal"

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
