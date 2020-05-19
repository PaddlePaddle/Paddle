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
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import core


class TestReverseOp(OpTest):
    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0]

    def setUp(self):
        self.initTestCase()
        self.op_type = "reverse"
        self.inputs = {"X": self.x}
        self.attrs = {'axis': self.axis}
        out = self.x
        for a in self.axis:
            out = np.flip(out, axis=a)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase0(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [1]


class TestCase0_neg(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [-1]


class TestCase1(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, 1]


class TestCase1_neg(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, -1]


class TestCase2(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, 2]


class TestCase2_neg(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, -2]


class TestCase3(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [1, 2]


class TestCase3_neg(TestReverseOp):
    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [-1, -2]


class TestCase4(unittest.TestCase):
    def test_error(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(
                name="label", shape=[1, 1, 1, 1, 1, 1, 1, 1], dtype="int64")
            rev = fluid.layers.reverse(label, axis=[-1, -2])

        def _run_program():
            x = np.random.random(size=(10, 1, 1, 1, 1, 1, 1)).astype('int64')
            exe.run(train_program, feed={"label": x})

        self.assertRaises(core.EnforceNotMet, _run_program)


if __name__ == '__main__':
    unittest.main()
