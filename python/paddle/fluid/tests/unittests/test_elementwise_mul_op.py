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
import unittest
import numpy as np
from op_test import OpTest


class ElementwiseMulOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestElementwiseMulOp_scalar(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(1).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_Vector(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, )).astype("float64"),
            'Y': np.random.random((32, )).astype("float64")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(2).astype(np.float64)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(2, 1, 1)
        }


class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(3).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 1)
        }


class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(4).astype(np.float64)
        }

        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 4)
        }


class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(np.float64),
            'Y': np.random.rand(3, 4).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 4, 1)
        }


if __name__ == '__main__':
    unittest.main()
