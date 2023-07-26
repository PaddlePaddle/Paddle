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


class TestReduceOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.dim = [0, 1]
        self.keepdim = False

    def set_op_type(self):
        return "reduce_sum"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"dim": self.dim, "keep_dim": self.keepdim}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestReduceSum(TestReduceOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.dim = [0]
        self.keepdim = False


class TestReduceSumCase1(TestReduceOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.dim = [0]
        self.keepdim = True


class TestReduceMax(TestReduceOp):
    def set_op_type(self):
        return "reduce_max"


class TestReduceMin(TestReduceOp):
    def set_op_type(self):
        return "reduce_min"


class TestReduceProd(TestReduceOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([2, 3], "float32", 1.0, 2.0)}
        self.dim = [0, 1]
        self.keepdim = False

    def set_op_type(self):
        return "reduce_prod"


class TestReduceMean(TestReduceOp):
    def set_op_type(self):
        return "reduce_mean"


class TestReduceMeanCase1(TestReduceMean):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", 1.0, 2.0)}
        self.dim = [1]
        self.keepdim = False


class TestReduceMeanCase2(TestReduceMean):
    def init_input_data(self):
        self.feed_data = {'x': self.random([16, 32, 64], "float32", 1.0, 2.0)}
        self.dim = [0, 1]
        self.keepdim = True


class TestReduceAll(TestReduceOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "bool")}
        self.dim = [0, 1]
        self.keepdim = False

    def set_op_type(self):
        return "reduce_all"


class TestReduceAny(TestReduceOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "bool")}
        self.dim = [0, 1]
        self.keepdim = False

    def set_op_type(self):
        return "reduce_any"


class TestReduceOutType(TestReduceOp):
    def set_op_attrs(self):
        return {
            "dim": self.dim,
            "keep_dim": self.keepdim,
            "out_dtype": self.nptype2paddledtype("float64"),
        }


class TestReduceUnkOutType(TestReduceOp):
    def set_op_attrs(self):
        return {
            "dim": self.dim,
            "keep_dim": self.keepdim,
            "out_dtype": self.nptype2paddledtype("unk"),
        }


if __name__ == "__main__":
    unittest.main()
