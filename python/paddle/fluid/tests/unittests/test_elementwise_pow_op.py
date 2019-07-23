#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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


class TestElementwisePowOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        print("====test_check_output====")
        self.check_output()
    def test_check_grad_normal(self):
        print("====test_check_grad_normal====")
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)


class TestElementwisePowOp_scalar(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype('float32'),
            'Y': np.random.rand(1).astype('float32')
        }
        print("!!!!input:\t",self.inputs['X'])
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}
        print("=====output:\t",self.outputs)

class TestElementwisePowOp_tensor(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.random((32, )).astype("float64"),
            'Y': np.random.random((32, )).astype("float64")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

if __name__ == '__main__':
    unittest.main()

