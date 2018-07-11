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


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.float32
        self.init_dtype()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


class TestElementwiseSubOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"

        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(self.dtype),
            'Y': np.random.rand(1).astype(self.dtype)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.random((32, )).astype(self.dtype),
            'Y': np.random.random((32, )).astype(self.dtype)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(self.dtype),
            'Y': np.random.rand(2).astype(self.dtype)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(2, 1, 1)
        }


class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(self.dtype),
            'Y': np.random.rand(3).astype(self.dtype)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 3, 1)
        }


class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(self.dtype),
            'Y': np.random.rand(4).astype(self.dtype)
        }

        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 4)
        }


class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(self.dtype),
            'Y': np.random.rand(3, 4).astype(self.dtype)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 3, 4, 1)
        }


class TestFP16ElementwiseSubOp_scalar(TestElementwiseSubOp_scalar):
    def init_dtype(self):
        self.dtype = np.float16


class TestFP16TestElementwiseSubOp_Vector(TestElementwiseSubOp_Vector):
    def init_dtype(self):
        self.dtype = np.float16


class TestFP16ElementwiseSubOp_broadcast_0(TestElementwiseSubOp_broadcast_0):
    def init_dtype(self):
        self.dtype = np.float16


class TestFP16ElementwiseSubOp_broadcast_1(TestElementwiseSubOp_broadcast_1):
    def init_dtype(self):
        self.dtype = np.float16


class TestFP16ElementwiseSubOp_broadcast_3(TestElementwiseSubOp_broadcast_3):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
