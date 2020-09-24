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
from op_test import OpTest, skip_check_grad_ci


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseSubOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float64),
            'Y': np.random.rand(1).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.random((100, )).astype("float64"),
            'Y': np.random.random((100, )).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }


class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64)
        }

        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float64),
            'Y': np.random.rand(10, 12).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(np.float64),
            'Y': np.random.rand(2, 5, 1, 12).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(1, 1, 100).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(np.float64),
            'Y': np.random.rand(10, 1, 12, 1).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 12).astype(np.float64),
            'Y': np.random.rand(2, 3, 10, 12).astype(np.float64)
        }

        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
        }


if __name__ == '__main__':
    unittest.main()
