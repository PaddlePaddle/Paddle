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
            'X': np.random.uniform(0.1, 1, [2, 3]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 3]).astype("float32")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestElementwisePowOp_scalar(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [3, 3, 4]).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, [1]).astype(np.float32)
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_tensor(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [32]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [32]).astype("float32")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [4]).astype("float32")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_broadcast_1(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [3]).astype("float32")
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.power(self.inputs['X'], self.inputs['Y'].reshape(3, 1))
        }


class TestElementwisePowOp_broadcast_2(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2]).astype("float32")
        }
        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': np.power(self.inputs['X'], self.inputs['Y'].reshape(2, 1, 1))
        }


class TestElementwisePowOp_broadcast_3(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [3, 4]).astype("float32")
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.power(self.inputs['X'], self.inputs['Y'].reshape(1, 3, 4,
                                                                       1))
        }


if __name__ == '__main__':
    unittest.main()
