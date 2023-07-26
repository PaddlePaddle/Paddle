#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

from op_mapper_test import OpMapperTest

import paddle


class TestCompareOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32", -10, 10),
            'y': self.random([32, 64], "float32", -10, 10),
        }

    def set_op_type(self):
        return "equal"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        y = paddle.static.data(
            name='y',
            shape=self.feed_data['y'].shape,
            dtype=self.feed_data['y'].dtype,
        )
        return {'X': [x], 'Y': [y]}

    def set_op_attrs(self):
        return {"axis": -1}

    def set_op_outputs(self):
        return {'Out': ['bool']}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestEqualOp(TestCompareOp):
    def set_op_type(self):
        return "equal"


class TestNotEqualOp(TestCompareOp):
    def set_op_type(self):
        return "not_equal"


class TestGreaterEqualOp(TestCompareOp):
    def set_op_type(self):
        return "greater_equal"


class TestGreaterThanOp(TestCompareOp):
    def set_op_type(self):
        return "greater_than"


class TestLessEqualOp(TestCompareOp):
    def set_op_type(self):
        return "less_equal"


class TestLessThanOp(TestCompareOp):
    def set_op_type(self):
        return "less_than"


class TestAxisCase(TestCompareOp):
    def set_op_attrs(self):
        return {"axis": 0}


if __name__ == "__main__":
    unittest.main()
