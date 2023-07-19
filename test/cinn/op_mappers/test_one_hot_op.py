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


class TestOneHotOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([1, 32], 'int32', low=0, high=9)}
        self.depth = 10
        self.dtype = "float32"
        self.allow_out_of_range = False

    def set_op_type(self):
        return "one_hot"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "depth": self.depth,
            "dtype": self.nptype2paddledtype(self.dtype),
            "allow_out_of_range": self.allow_out_of_range,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestOneHotOpCase1(TestOneHotOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], 'int32')}
        self.depth = 64
        self.dtype = "int32"
        self.allow_out_of_range = False


class TestOneHotOpCase2(TestOneHotOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64, 1], 'int64')}
        self.depth = 1
        self.dtype = "int64"
        self.allow_out_of_range = True


class TestOneHotV2Op(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([1, 32], 'int32')}
        self.depth = 10
        self.dtype = "float32"
        self.allow_out_of_range = False

    def set_op_type(self):
        return "one_hot_v2"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "depth": self.depth,
            "dtype": self.nptype2paddledtype(self.dtype),
            "allow_out_of_range": self.allow_out_of_range,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestOneHotV2OpCase1(TestOneHotV2Op):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], 'int32')}
        self.depth = 64
        self.dtype = "int32"
        self.allow_out_of_range = False


class TestOneHotV2OpCase2(TestOneHotV2Op):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64, 1], 'int64')}
        self.depth = 1
        self.dtype = "int64"
        self.allow_out_of_range = True


if __name__ == "__main__":
    unittest.main()
