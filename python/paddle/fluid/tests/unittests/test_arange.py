#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid import core
from paddle.static import program_guard, Program
import unittest
import numpy as np
from op_test import OpTest


class TestArangeOp(OpTest):

    def setUp(self):
        self.op_type = "range"
        self.init_config()
        self.inputs = {
            'Start': np.array([self.case[0]]).astype(self.dtype),
            'End': np.array([self.case[1]]).astype(self.dtype),
            'Step': np.array([self.case[2]]).astype(self.dtype)
        }

        self.outputs = {
            'Out':
            np.arange(self.case[0], self.case[1],
                      self.case[2]).astype(self.dtype)
        }

    def init_config(self):
        self.dtype = np.float32
        self.case = (0, 1, 0.2)

    def test_check_output(self):
        self.check_output()


class TestFloatArangeOp(TestArangeOp):

    def init_config(self):
        self.dtype = np.float32
        self.case = (0, 5, 1)


class TestInt32ArangeOp(TestArangeOp):

    def init_config(self):
        self.dtype = np.int32
        self.case = (0, 5, 2)


class TestFloat64ArangeOp(TestArangeOp):

    def init_config(self):
        self.dtype = np.float64
        self.case = (10, 1, -2)


class TestInt64ArangeOp(TestArangeOp):

    def init_config(self):
        self.dtype = np.int64
        self.case = (-1, -10, -2)


class TestArangeOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            self.assertRaises(TypeError, paddle.arange, 10, dtype='int8')


class TestArangeAPI(unittest.TestCase):

    def test_out(self):
        with program_guard(Program(), Program()):
            x1 = paddle.arange(0, 5, 1, 'float32')

            place = paddle.CUDAPlace(
                0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            out = exe.run(fetch_list=[x1])

        expected_data = np.arange(0, 5, 1).astype(np.float32)
        self.assertEqual((out == expected_data).all(), True)


class TestArangeImperative(unittest.TestCase):

    def test_out(self):
        place = paddle.CUDAPlace(
            0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        paddle.disable_static(place)
        x1 = paddle.arange(0, 5, 1)
        x2 = paddle.tensor.arange(5)
        x3 = paddle.tensor.creation.arange(5)

        start = paddle.to_tensor(np.array([0], 'float32'))
        end = paddle.to_tensor(np.array([5], 'float32'))
        step = paddle.to_tensor(np.array([1], 'float32'))
        x4 = paddle.arange(start, end, step, 'int64')
        paddle.enable_static()

        expected_data = np.arange(0, 5, 1).astype(np.int64)
        for i in [x1, x2, x3, x4]:
            self.assertEqual((i.numpy() == expected_data).all(), True)


if __name__ == "__main__":
    unittest.main()
