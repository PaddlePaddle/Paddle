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
import sys

sys.path.append("..")
from op_test import OpTest
import paddle

paddle.enable_static()


class TestMLUReduceSumOp(OpTest):

    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.set_mlu()
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keep_dim,
            'reduce_all': self.reduce_all
        }
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        if self.attrs['reduce_all']:
            self.outputs = {'Out': self.inputs['X'].sum()}
        else:
            self.outputs = {
                'Out':
                self.inputs['X'].sum(axis=self.axis,
                                     keepdims=self.attrs['keep_dim'])
            }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6, 10)
        self.axis = (0, )


class TestSumOp5D(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (1, 2, 5, 6, 10)
        self.axis = (0, )


class TestSumOp6D(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (1, 1, 2, 5, 6, 10)
        self.axis = (0, )


class TestSumOp8D(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (1, 3, 1, 2, 1, 4, 3, 10)
        self.axis = (0, 3)


class Test1DReduce(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = 120
        self.axis = (0, )


class Test2DReduce0(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (20, 10)
        self.axis = (0, )


class Test2DReduce1(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (20, 10)
        self.axis = (1, )


class Test3DReduce0(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 7)
        self.axis = (1, )


class Test3DReduce1(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 7)
        self.axis = (2, )


class Test3DReduce2(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 7)
        self.axis = (-2, )


class Test3DReduce3(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 7)
        self.axis = (1, 2)


class TestKeepDimReduce(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 10)
        self.axis = (1, )
        self.keep_dim = True


class TestKeepDim8DReduce(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (2, 5, 3, 2, 2, 3, 4, 2)
        self.axis = (3, 4, 5)
        self.keep_dim = True

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=0.03)


class TestReduceAll(TestMLUReduceSumOp):

    def initTestCase(self):
        self.shape = (5, 6, 2, 10)
        self.axis = (0, )
        self.reduce_all = True


if __name__ == '__main__':
    unittest.main()
