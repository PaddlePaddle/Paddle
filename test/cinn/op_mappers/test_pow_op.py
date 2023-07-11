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


class TestPowOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
            'factor': self.random([1], "float32", 0.0, 4.0),
        }

    def set_op_type(self):
        return "pow"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        factor = paddle.static.data(
            name='factor',
            shape=self.feed_data['factor'].shape,
            dtype=self.feed_data['factor'].dtype,
        )
        return {'X': [x], 'FactorTensor': [factor]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestPowCase1(TestPowOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", 2, 10),
            'factor': self.random([1], "int32", 0, 5),
        }


class TestPowCase2(TestPowOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", 2, 10),
            'factor': self.random([1], "int32", 0, 5),
        }


class TestPowOpInFactorAttr(TestPowOp):
    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"factor": float(2)}


if __name__ == "__main__":
    unittest.main()
