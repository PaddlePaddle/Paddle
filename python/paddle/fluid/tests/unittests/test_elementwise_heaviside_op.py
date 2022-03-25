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
from op_test import OpTest
import paddle

paddle.enable_static()


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((13, 17)).astype("float64")
        y = np.random.random((13, 17)).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestElementwiseHeavisideOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((100, )).astype("float64")
        y = np.random.random((100, )).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((100, 5, 2)).astype(np.float64)
        y = np.random.random((100, 1, 1)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((2, 100, 3)).astype(np.float64)
        y = np.random.random((100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(
                1, 100, 1))
        }


class TestElementwiseHeavisideOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((1, 3, 100)).astype(np.float64)
        y = np.random.random((100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out':
            np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))
        }


class TestElementwiseHeavisideOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((2, 50, 2, 1)).astype(np.float64)
        y = np.random.random((50, 2)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(1, 50, 2, 1))
        }


class TestElementwiseHeavisideOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((2, 3, 4, 50)).astype(np.float64)
        y = np.random.random((2, 3, 1, 50)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
