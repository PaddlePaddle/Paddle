#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestFillConstantOp1(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp2(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92]}
        self.outputs = {'Out': np.full((123, 92), 0.0)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp3(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified int64 value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 10000000000}
        self.outputs = {'Out': np.full((123, 92), 10000000000)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp4(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified int value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3}
        self.outputs = {'Out': np.full((123, 92), 3)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOpWithSelectedRows(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()
        # create Out Variable
        out = scope.var('Out').get_selected_rows()

        # create and run fill_constant_op operator
        fill_constant_op = Operator(
            "fill_constant", shape=[123, 92], value=3.8, Out='Out')
        fill_constant_op.run(scope, place)

        # get result from Out
        result_array = np.array(out.get_tensor())
        full_array = np.full((123, 92), 3.8, 'float32')

        self.assertTrue(np.array_equal(result_array, full_array))

    def test_fill_constant_with_selected_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
