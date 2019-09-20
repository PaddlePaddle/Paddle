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


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')
        self.check_grad(['x1'], 'Out')
        self.check_grad(['x2'], 'Out')

    def init_test_data(self):
        self.x0 = np.random.random((2, 1, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 2, 4, 5)).astype('float32')
        self.x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.axis = 1


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.axis = 1


class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 256, 170, 256)).astype('float32')
        self.x1 = np.random.random((1, 128, 170, 256)).astype('float32')
        self.x2 = np.random.random((1, 128, 170, 256)).astype('float32')
        self.axis = 1

    def test_check_grad(self):
        pass


class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x2 = np.random.random((0, 3, 4, 5)).astype('float32')
        self.axis = 0

    def test_check_grad(self):
        pass


class TestConcatOp5(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 1, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 2, 4, 5)).astype('float32')
        self.x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.axis = -3


if __name__ == '__main__':
    unittest.main()
