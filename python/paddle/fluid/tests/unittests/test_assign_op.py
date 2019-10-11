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

import op_test
import numpy as np
import unittest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestAssignOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign"
        x = numpy.random.random(size=(100, 10))
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out')


class TestAssignOpError(op_test.OpTest):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.assign, x1)
            # When the type of input is Variable, the dtype of input must be float32, float64, int32, int64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="float16")
            self.assertRaises(TypeError, fluid.layers.assign, x2)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="int8")
            self.assertRaises(TypeError, fluid.layers.assign, x3)
            x4 = fluid.layers.data(name='x4', shape=[4], dtype="int16")
            self.assertRaises(TypeError, fluid.layers.assign, x4)
            x5 = fluid.layers.data(name='x5', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.assign, x5)
            x6 = fluid.layers.data(name='x6', shape=[4], dtype="uint16")
            self.assertRaises(TypeError, fluid.layers.assign, x6)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x7 = np.array([[2.5, 2.5]], dtype='float16')
            self.assertRaises(TypeError, fluid.layers.assign, x7)
            x8 = np.array([[2.5, 2.5]], dtype='float64')
            self.assertRaises(TypeError, fluid.layers.assign, x8)
            x9 = np.array([[2.5, 2.5]], dtype='int8')
            self.assertRaises(TypeError, fluid.layers.assign, x9)
            x10 = np.array([[2.5, 2.5]], dtype='int16')
            self.assertRaises(TypeError, fluid.layers.assign, x10)
            x11 = np.array([[2.5, 2.5]], dtype='int64')
            self.assertRaises(TypeError, fluid.layers.assign, x11)
            x12 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, fluid.layers.assign, x12)
            x13 = np.array([[2.5, 2.5]], dtype='uint16')
            self.assertRaises(TypeError, fluid.layers.assign, x13)
            x14 = np.array([[2.5, 2.5]], dtype='uint64')
            self.assertRaises(TypeError, fluid.layers.assign, x14)


if __name__ == '__main__':
    unittest.main()
