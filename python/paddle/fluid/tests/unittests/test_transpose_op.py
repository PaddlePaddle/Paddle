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
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestTransposeOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {
            'XShape': np.random.random(self.shape).astype("float32"),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = False

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (3, 40)
        self.axis = (1, 0)


class TestCase0(TestTransposeOp):
    def initTestCase(self):
        self.shape = (100, )
        self.axis = (0, )


class TestCase1(TestTransposeOp):
    def initTestCase(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase2(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestTransposeOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[10, 5, 3], dtype='float32')

            def test_x_Variable_check():
                # the Input(x)'s type must be Variable
                fluid.layers.transpose("not_variable", perm=[1, 0, 2])

            self.assertRaises(TypeError, test_x_Variable_check)

            def test_x_dtype_check():
                # the Input(x)'s dtype must be one of [float16, float32, float64, int32, int64]
                x1 = fluid.layers.data(
                    name='x1', shape=[10, 5, 3], dtype='bool')
                fluid.layers.transpose(x1, perm=[1, 0, 2])

            self.assertRaises(TypeError, test_x_dtype_check)

            def test_perm_list_check():
                # Input(perm)'s type must be list
                fluid.layers.transpose(x, perm="[1, 0, 2]")

            self.assertRaises(TypeError, test_perm_list_check)

            def test_perm_length_and_x_dim_check():
                # Input(perm) is the permutation of dimensions of Input(input)
                # its length should be equal to dimensions of Input(input)
                fluid.layers.transpose(x, perm=[1, 0, 2, 3, 4])

            self.assertRaises(ValueError, test_perm_length_and_x_dim_check)

            def test_each_elem_value_check():
                # Each element in Input(perm) should be less than Input(x)'s dimension
                fluid.layers.transpose(x, perm=[3, 5, 7])

            self.assertRaises(ValueError, test_each_elem_value_check)


if __name__ == '__main__':
    unittest.main()
