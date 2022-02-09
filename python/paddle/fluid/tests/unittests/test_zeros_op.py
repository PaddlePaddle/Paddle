# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import _test_eager_guard


class TestZerosOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            shape = [4]
            dtype = 'int8'
            self.assertRaises(TypeError, fluid.layers.zeros, shape, dtype)

    def test_eager(self):
        with _test_eager_guard():
            self.test_errors()


class ApiZerosTest(unittest.TestCase):
    def test_out(self):
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='float64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result, ) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='float64')
        self.assertEqual((result == expected_result).all(), True)
        with paddle.static.program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result, ) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result, ) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            out_np = np.zeros(shape=1, dtype='float32')
            out = paddle.zeros(shape=[1], dtype='float32')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(fetch_list=[out])
            self.assertEqual((result == out_np).all(), True)

    def test_fluid_out(self):
        with program_guard(Program()):
            zeros = fluid.layers.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result, ) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)

    def test_eager(self):
        with _test_eager_guard():
            self.test_out()
            self.test_fluid_out()


class ApiZerosError(unittest.TestCase):
    def test_errors(self):
        def test_error1():
            with paddle.static.program_guard(fluid.Program()):
                ones = fluid.layers.zeros(shape=10, dtype='int64')

        self.assertRaises(TypeError, test_error1)

        def test_error2():
            with paddle.static.program_guard(fluid.Program()):
                ones = fluid.layers.zeros(shape=[10], dtype='int8')

        self.assertRaises(TypeError, test_error2)

    def test_eager(self):
        with _test_eager_guard():
            self.test_errors()


if (__name__ == '__main__'):
    unittest.main()
