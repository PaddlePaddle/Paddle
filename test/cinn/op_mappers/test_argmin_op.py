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


class TestArgminOp(OpMapperTest):
    def init_input_data(self):
        self.axis = 1
        self.shape = [2, 3, 4]
        self.input_dtype = "float32"
        self.output_dtype = "int64"
        self.flatten = False
        self.keepdims = False
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }

    def set_op_type(self):
        return "arg_min"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "axis": self.axis,
            "flatten": self.flatten,
            "keepdims": self.keepdims,
            "dtype": self.nptype2paddledtype(self.output_dtype),
        }

    def set_op_outputs(self):
        return {'Out': [str(self.output_dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestArgminCase1(TestArgminOp):
    """
    Test case with negative axis and True flatten and int64 output dtype
    """

    def init_input_data(self):
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = "float32"
        self.output_dtype = "int64"
        self.keepdims = False
        self.flatten = True
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class TestArgminCase2(TestArgminOp):
    """
    Test case with true keepdims
    """

    def init_input_data(self):
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = "float32"
        self.output_dtype = "int32"
        self.flatten = False
        self.keepdims = True
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class TestArgminCase3(TestArgminOp):
    """
    Test case with input_dtype float64
    """

    def init_input_data(self):
        self.axis = -1
        self.shape = [2, 3, 4]
        self.input_dtype = "float64"
        self.output_dtype = "int32"
        self.flatten = False
        self.keepdims = True
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


if __name__ == "__main__":
    unittest.main()
