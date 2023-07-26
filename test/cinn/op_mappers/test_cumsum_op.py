#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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


class TestCumsumOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], "float32"),
        }

    def set_op_type(self):
        return "cumsum"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"axis": -1, "flatten": False}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestCumsumCase1(TestCumsumOp):
    """
    Test case with negative axis
    """

    def set_op_attrs(self):
        return {"axis": -3, "flatten": False}


class TestCumsumCase2(TestCumsumOp):
    """
    Test case with unspecified axis and dtype
    flatten = True if axis is None, and axis can ignore
    """

    def set_op_attrs(self):
        return {"flatten": True}


class TestCumsumCase3(TestCumsumOp):
    """
    Test case with dtype int32
    """

    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], "int32", -10, 10),
        }

    def set_op_attrs(self):
        return {"axis": 1, "flatten": False}


class TestCumsumCase4(TestCumsumOp):
    """
    Test case with dtype int64
    """

    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3], "int64", -10, 10),
        }


class TestCumsumCase5(TestCumsumOp):
    """
    Test case with exclusive = True
    """

    def set_op_attrs(self):
        return {"exclusive": True}


class TestCumsumCase6(TestCumsumOp):
    """
    Test case with reverse = True
    """

    def set_op_attrs(self):
        return {"reverse": True}


if __name__ == "__main__":
    unittest.main()
