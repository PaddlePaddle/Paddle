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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("../")
from op_test import OpTest

import paddle
from paddle import fluid


class TestSequenceConcat(OpTest):
    def setLoD(self):
        self.lod1 = [7, 3]
        self.lod2 = [12, 8]
        self.out_lod = [19, 11]

    def setUp(self):
        x1 = np.random.random(size=(10, 80)).astype('float64')
        x2 = np.random.random(size=(20, 80)).astype('float64')
        self.setLoD()

        out = np.concatenate((x1[0:self.lod1[0]], x2[0:self.lod2[0]],
                              x1[self.lod1[0]:], x2[self.lod2[0]:]))

        self.op_type = "sequence_concat"
        self.inputs = {
            'X': [("x1", (x1, [self.lod1])), ("x2", (x2, [self.lod2]))]
        }
        self.outputs = {"Out": (out, [self.out_lod])}

    def test_output(self):
        self.check_output()

    def test_dx(self):
        self.check_grad(inputs_to_check=['x1', 'x2'], output_names="Out")


class TestSequenceConcatCase2(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [10, 0]
        self.lod2 = [12, 8]
        self.out_lod = [22, 8]


class TestSequenceConcatCase3(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [10, 0]
        self.lod2 = [20, 0]
        self.out_lod = [30, 0]


class TestSequenceConcatCase4(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [0, 10]
        self.lod2 = [0, 20]
        self.out_lod = [0, 30]


class TestSequenceConcatCase5(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [0, 10]
        self.lod2 = [20, 0]
        self.out_lod = [20, 10]


class TestSequenceConcatOpError(unittest.TestCase):
    def test_errors(self):
        def test_input_list():
            # the input type must be list
            x_data = fluid.layers.data(name='x', shape=[4], dtype='float32')
            fluid.layers.sequence_concat(input=x_data)

        self.assertRaises(TypeError, test_input_list)

        def test_variable1():
            # the input element type must be Variable
            x1_data = np.array([[3, 5]]).astype('float32')
            y1_data = fluid.layers.data(name='y1', shape=[4], dtype='float32')
            fluid.layers.sequence_concat(input=[x1_data, y1_data])

        def test_variable2():
            x2_data = np.array([[3, 5]]).astype('float32')
            y2_data = fluid.layers.data(name='y2', shape=[4], dtype='float32')
            fluid.layers.sequence_concat(input=[y2_data, x2_data])

        for i in range(2):
            if i == 0:
                self.assertRaises(TypeError, test_variable1)
            else:
                self.assertRaises(TypeError, test_variable2)

        def test_dtype():
            # dtype must be 'float32', 'float64', 'int64'
            x3_data = fluid.layers.data(name="x3", shape=[3, 5], dtype='int32')
            y3_data = fluid.layers.data(name="y3", shape=[3, 5], dtype='int16')
            input_list = [x3_data, y3_data]
            fluid.layers.sequence_concat(input=input_list)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
