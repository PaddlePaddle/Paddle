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

from op_test import OpTest
import numpy as np
import unittest


class TestUnStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        pass

    def get_y_names(self):
        y_names = []
        for i in range(self.input_dim[self.axis]):
            y_names.append('y{}'.format(i))
        return y_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'unstack'
        self.x = np.random.random(size=self.input_dim).astype(self.dtype)

        outs = np.split(self.x, self.input_dim[self.axis], self.axis)
        new_shape = list(self.input_dim)
        del new_shape[self.axis]
        y_names = self.get_y_names()
        tmp = []
        for i in range(self.input_dim[self.axis]):
            tmp.append((y_names[i], np.reshape(outs[i], new_shape)))

        self.inputs = {'X': self.x}
        self.outputs = {'Y': tmp}
        self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], self.get_y_names())


class TestStackOp3(TestUnStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestUnStackOpBase):
    def initParameters(self):
        self.axis = -3


class TestStackOp5(TestUnStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestUnStackOpBase):
    def initParameters(self):
        self.axis = 2


if __name__ == '__main__':
    unittest.main()
