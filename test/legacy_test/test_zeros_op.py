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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import Program, program_guard


class ApiZerosTest(unittest.TestCase):
    def test_out(self):
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='float64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='float64')
        self.assertEqual((result == expected_result).all(), True)
        with paddle.static.program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int8')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int8')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            out_np = np.zeros(shape=1, dtype='float32')
            out = paddle.zeros(shape=[1], dtype='float32')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(fetch_list=[out])
            self.assertEqual((result == out_np).all(), True)
        with program_guard(Program()):
            out_np = np.zeros(shape=10, dtype='int32')
            out = paddle.zeros(shape=10, dtype='int32')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(fetch_list=[out])
            self.assertEqual((result == out_np).all(), True)


class ApiZerosError(unittest.TestCase):
    def test_shape_errors(self):
        with base.dygraph.guard():
            try:
                shape = [-1, 5]
                out = paddle.zeros(shape)
            except Exception as e:
                error_msg = str(e)
                assert error_msg.find("expected to be no less than 0") > 0


class ApiZerosWithDynamicShape(unittest.TestCase):
    def test_dynamic_shape(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data("x", shape=[], dtype='int32')
            out = paddle.zeros(shape=[101, x])
            self.assertEqual(out.shape, [101, -1])


if __name__ == '__main__':
    unittest.main()
