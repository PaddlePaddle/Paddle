# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import base


class TestTriuIndicesOp(OpTest):
    def setUp(self):
        self.op_type = "triu_indices"
        self.python_api = paddle.triu_indices
        self.inputs = {}
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def init_config(self):
        self.attrs = {'row': 4, 'col': 4, 'offset': -1}
        self.target = np.triu_indices(
            self.attrs['row'], self.attrs['offset'], self.attrs['col']
        )
        self.target = np.array(self.target)


class TestTriuIndicesOpCase1(TestTriuIndicesOp):
    def init_config(self):
        self.attrs = {'row': 0, 'col': 0, 'offset': 0}
        self.target = np.triu_indices(0, 0, 0)
        self.target = np.array(self.target)


class TestTriuIndicesOpCase2(TestTriuIndicesOp):
    def init_config(self):
        self.attrs = {'row': 4, 'col': 4, 'offset': 2}
        self.target = np.triu_indices(
            self.attrs['row'], self.attrs['offset'], self.attrs['col']
        )
        self.target = np.array(self.target)


class TestTriuIndicesAPICaseStatic(unittest.TestCase):

    def test_static(self):
        if base.core.is_compiled_with_cuda():
            place = paddle.base.CUDAPlace(0)
        else:
            place = paddle.CPUPlace()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data = paddle.triu_indices(4, 4, -1)
            exe = paddle.static.Executor(place)
            result = exe.run(feed={}, fetch_list=[data])
        expected_result = np.triu_indices(4, -1, 4)
        np.testing.assert_array_equal(result[0], expected_result)


class TestTriuIndicesAPICaseDygraph(unittest.TestCase):
    def test_dygraph(self):
        if base.core.is_compiled_with_cuda():
            place = paddle.base.CUDAPlace(0)
        else:
            place = paddle.CPUPlace()
        with base.dygraph.base.guard(place=place):
            out = paddle.triu_indices(4, 4, 2)
        expected_result = np.triu_indices(4, 2, 4)
        np.testing.assert_array_equal(out, expected_result)


class TestTriuIndicesAPICaseError(unittest.TestCase):
    def test_case_error(self):
        def test_num_rows_type_check():
            out1 = paddle.triu_indices(1.0, 1, 2)

        self.assertRaises(TypeError, test_num_rows_type_check)

        def test_num_columns_type_check():
            out2 = paddle.triu_indices(4, -1, 2)

        self.assertRaises(TypeError, test_num_columns_type_check)

        def test_num_offset_type_check():
            out3 = paddle.triu_indices(4, 4, 2.0)

        self.assertRaises(TypeError, test_num_offset_type_check)


class TestTriuIndicesAPICaseDefault(unittest.TestCase):

    def test_default_CPU(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data = paddle.triu_indices(4, None, 2)
            exe = paddle.static.Executor(paddle.CPUPlace())
            result = exe.run(feed={}, fetch_list=[data])
        expected_result = np.triu_indices(4, 2)
        np.testing.assert_array_equal(result[0], expected_result)

        with base.dygraph.base.guard(paddle.CPUPlace()):
            out = paddle.triu_indices(4, None, 2)
        expected_result = np.triu_indices(4, 2)
        np.testing.assert_array_equal(out, expected_result)


if __name__ == "__main__":
    unittest.main()
