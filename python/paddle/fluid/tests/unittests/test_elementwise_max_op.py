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
from op_test import OpTest


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_max"
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float32")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}

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


class TestElementwiseMaxOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.random_integers(-5, 5, [2, 3, 4]).astype("float32")
        y = np.array([0.5]).astype("float32")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.random((32, )).astype("float32")
        sgn = np.random.choice([-1, 1], (32, )).astype("float32")
        y = x + sgn * np.random.uniform(0.1, 1, (32, )).astype("float32")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.uniform(0.5, 1, (2, 3, 4)).astype(np.float32)
        sgn = np.random.choice([-1, 1], (2, )).astype(np.float32)
        y = x[:, 0, 0] + sgn * \
            np.random.uniform(1, 2, (2, )).astype(np.float32)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(2, 1, 1))
        }


class TestElementwiseMaxOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.uniform(0.5, 1, (2, 3, 4)).astype(np.float32)
        sgn = np.random.choice([-1, 1], (3, )).astype(np.float32)
        y = x[0, :, 0] + sgn * \
            np.random.uniform(1, 2, (3, )).astype(np.float32)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(1, 3, 1))
        }


class TestElementwiseMaxOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.uniform(0.5, 1, (2, 3, 4)).astype(np.float32)
        sgn = np.random.choice([-1, 1], (4, )).astype(np.float32)
        y = x[0, 0, :] + sgn * \
            np.random.uniform(1, 2, (4, )).astype(np.float32)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 4))
        }


class TestElementwiseMaxOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(np.float32)
        sgn = np.random.choice([-1, 1], (3, 4)).astype(np.float32)
        y = x[0, :, :, 0] + sgn * \
            np.random.uniform(1, 2, (3, 4)).astype(np.float32)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(1, 3, 4, 1))
        }


if __name__ == '__main__':
    unittest.main()
