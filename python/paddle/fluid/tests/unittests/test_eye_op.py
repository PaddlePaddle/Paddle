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
        with fluid.program_guard(fluid.Program()):
            data = paddle.eye(10)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = paddle.eye(10, num_columns=7, dtype="float64")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, 7, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = paddle.eye(10, dtype="int64")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

    def test_errors(self):
        with fluid.program_guard(fluid.Program()):

            def test_num_rows_type_check():
                paddle.eye(-1, dtype="int64")

            self.assertRaises(TypeError, test_num_rows_type_check)

            def test_num_columns_type_check():
                paddle.eye(10, num_columns=5.2, dtype="int64")

            self.assertRaises(TypeError, test_num_columns_type_check)


if __name__ == "__main__":
    unittest.main()
