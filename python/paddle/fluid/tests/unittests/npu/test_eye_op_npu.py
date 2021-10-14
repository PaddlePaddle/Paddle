#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.fluid.framework as framework

paddle.enable_static()
np.random.seed(10)


class TestEyeOp(OpTest):
    def setUp(self):
        '''
	    Test eye op with specified shape
        '''
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "eye"
        self.inputs = {}

        self.num_rows = 0
        self.num_columns = 0
        self.dtype = np.float32

        self.initTestCase()

        if self.num_columns == 0:
            self.attrs = {
                'num_rows': self.num_rows,
                'dtype': framework.convert_np_dtype_to_dtype_(self.dtype)
            }
            self.outputs = {'Out': np.eye(self.num_rows, dtype=self.dtype)}
        else:
            self.attrs = {
                'num_rows': self.num_rows,
                'num_columns': self.num_columns,
                'dtype': framework.convert_np_dtype_to_dtype_(self.dtype)
            }
            self.outputs = {
                'Out': np.eye(self.num_rows, self.num_columns, dtype=self.dtype)
            }

    def initTestCase(self):
        self.num_rows = 219
        self.num_columns = 319
        self.dtype = np.int32

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestEyeOp1(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 50


class TestEyeOp2(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 50
        self.dtype = np.int32


class TestEyeOp3(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 50
        self.dtype = np.float16


class TestEyeOp4(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 1
        self.num_columns = 99


class TestEyeOp5(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 100
        self.num_columns = 100


class TestEyeOp6(TestEyeOp):
    def initTestCase(self):
        self.num_rows = 100
        self.num_columns = 100
        self.dtype = np.float32


class API_TestTensorEye(unittest.TestCase):
    def test_out(self):
        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10)
            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, num_columns=7, dtype="float16")
            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, 7, dtype="float16")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, dtype="int32")
            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="int32")
        self.assertEqual((result == expected_result).all(), True)

        paddle.disable_static(paddle.NPUPlace(0))
        out = paddle.eye(10, dtype="int32")
        expected_result = np.eye(10, dtype="int32")
        paddle.enable_static()
        self.assertEqual((out.numpy() == expected_result).all(), True)

        paddle.disable_static(paddle.NPUPlace(0))
        batch_shape = [2]
        out = fluid.layers.eye(10, 10, dtype="int32", batch_shape=batch_shape)
        result = np.eye(10, dtype="int32")
        expected_result = []
        for index in reversed(batch_shape):
            tmp_result = []
            for i in range(index):
                tmp_result.append(result)
            result = tmp_result
            expected_result = np.stack(result, axis=0)
        paddle.enable_static()
        self.assertEqual(out.numpy().shape == np.array(expected_result).shape,
                         True)
        self.assertEqual((out.numpy() == expected_result).all(), True)

        paddle.disable_static(paddle.NPUPlace(0))
        batch_shape = [3, 2]
        out = fluid.layers.eye(10, 10, dtype="int32", batch_shape=batch_shape)
        result = np.eye(10, dtype="int32")
        expected_result = []
        for index in reversed(batch_shape):
            tmp_result = []
            for i in range(index):
                tmp_result.append(result)
            result = tmp_result
            expected_result = np.stack(result, axis=0)
        paddle.enable_static()
        self.assertEqual(out.numpy().shape == np.array(expected_result).shape,
                         True)
        self.assertEqual((out.numpy() == expected_result).all(), True)

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):

            def test_num_rows_type_check():
                paddle.eye(-1, dtype="int64")

            self.assertRaises(TypeError, test_num_rows_type_check)

            def test_num_columns_type_check():
                paddle.eye(10, num_columns=5.2, dtype="int64")

            self.assertRaises(TypeError, test_num_columns_type_check)

            def test_num_columns_type_check1():
                paddle.eye(10, num_columns=10, dtype="int8")

            self.assertRaises(TypeError, test_num_columns_type_check1)


if __name__ == "__main__":
    unittest.main()
