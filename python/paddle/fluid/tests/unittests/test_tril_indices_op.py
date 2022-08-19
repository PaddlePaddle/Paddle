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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard


class TestTrilIndicesOp(OpTest):

    def setUp(self):
        self.op_type = "tril_indices"
        self.inputs = {}
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def init_config(self):
        self.attrs = {'rows': 4, 'cols': 4, 'offset': -1}
        self.target = np.tril_indices(self.attrs['rows'], self.attrs['offset'],
                                      self.attrs['cols'])
        self.target = np.array(self.target)


class TestTrilIndicesOpCase1(TestTrilIndicesOp):

    def init_config(self):
        self.attrs = {'rows': 0, 'cols': 0, 'offset': 0}
        self.target = np.tril_indices(0, 0, 0)
        self.target = np.array(self.target)


class TestTrilIndicesOpCase2(TestTrilIndicesOp):

    def init_config(self):
        self.attrs = {'rows': 4, 'cols': 4, 'offset': 2}
        self.target = np.tril_indices(self.attrs['rows'], self.attrs['offset'],
                                      self.attrs['cols'])
        self.target = np.array(self.target)


class TestTrilIndicesAPICaseStatic(unittest.TestCase):

    def test_static(self):
        places = [
            paddle.CPUPlace(), paddle.fluid.CUDAPlace(0)
        ] if fluid.core.is_compiled_with_cuda() else [paddle.CPUPlace()]
        paddle.enable_static()
        for place in places:
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                data1 = paddle.tril_indices(4, 4, -1)
                exe1 = paddle.static.Executor(place)
                result1, = exe1.run(feed={}, fetch_list=[data1])
            expected_result1 = np.tril_indices(4, -1, 4)
            np.testing.assert_allclose(result1, expected_result1, rtol=1e-05)


class TestTrilIndicesAPICaseDygraph(unittest.TestCase):

    def test_dygraph(self):
        places = [
            paddle.CPUPlace(), paddle.fluid.CUDAPlace(0)
        ] if fluid.core.is_compiled_with_cuda() else [paddle.CPUPlace()]
        for place in places:
            with fluid.dygraph.base.guard(place=place):
                out1 = paddle.tril_indices(4, 4, 2)
            expected_result1 = np.tril_indices(4, 2, 4)
            self.assertEqual((out1.numpy() == expected_result1).all(), True)

    def test_dygraph_eager(self):
        with _test_eager_guard():
            self.test_dygraph()


class TestTrilIndicesAPICaseError(unittest.TestCase):

    def test_case_error(self):

        def test_num_rows_type_check():
            out1 = paddle.tril_indices(1.0, 1, 2)

        self.assertRaises(TypeError, test_num_rows_type_check)

        def test_num_columns_type_check():
            out2 = paddle.tril_indices(4, -1, 2)

        self.assertRaises(TypeError, test_num_columns_type_check)

        def test_num_offset_type_check():
            out3 = paddle.tril_indices(4, 4, 2.0)

        self.assertRaises(TypeError, test_num_offset_type_check)


class TestTrilIndicesAPICaseDefault(unittest.TestCase):

    def test_default_CPU(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.tril_indices(4, None, 2)
            exe = paddle.static.Executor(paddle.CPUPlace())
            result, = exe.run(feed={}, fetch_list=[data])
        expected_result = np.tril_indices(4, 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-05)

        with fluid.dygraph.base.guard(paddle.CPUPlace()):
            out = paddle.tril_indices(4, None, 2)
        expected_result = np.tril_indices(4, 2)
        self.assertEqual((out.numpy() == expected_result).all(), True)


if __name__ == "__main__":
    unittest.main()
