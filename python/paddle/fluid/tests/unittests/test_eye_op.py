#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.framework as framework


class TestEyeOp(OpTest):
    def setUp(self):
        '''
	Test eye op with specified shape
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {
            'num_rows': 219,
            'num_columns': 319,
            'dtype': framework.convert_np_dtype_to_dtype_(np.int32)
        }
        self.outputs = {'Out': np.eye(219, 319, dtype=np.int32)}

    def test_check_output(self):
        self.check_output()


class TestEyeOp1(OpTest):
    def setUp(self):
        '''
	Test eye op with default parameters
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 50}
        self.outputs = {'Out': np.eye(50, dtype=float)}

    def test_check_output(self):
        self.check_output()


class TestEyeOp2(OpTest):
    def setUp(self):
        '''
        Test eye op with specified shape
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 99, 'num_columns': 1}
        self.outputs = {'Out': np.eye(99, 1, dtype=float)}

    def test_check_output(self):
        self.check_output()


class API_TestTensorEye(unittest.TestCase):
    def test_out(self):
        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10)
            place = fluid.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, num_columns=7, dtype="float64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, 7, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

        paddle.disable_static()
        out = paddle.eye(10, dtype="int64")
        expected_result = np.eye(10, dtype="int64")
        paddle.enable_static()
        self.assertEqual((out.numpy() == expected_result).all(), True)

        paddle.disable_static()
        batch_shape = [2]
        out = fluid.layers.eye(10, 10, dtype="int64", batch_shape=batch_shape)
        result = np.eye(10, dtype="int64")
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

        paddle.disable_static()
        batch_shape = [3, 2]
        out = fluid.layers.eye(10, 10, dtype="int64", batch_shape=batch_shape)
        result = np.eye(10, dtype="int64")
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
