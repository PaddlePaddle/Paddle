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


class TestAtan2Op(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
            'y': self.random([32, 64], "float32"),
        }

    def set_op_type(self):
        return "atan2"

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
        return {'X1': [x], 'X2': [y]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestAtan2OpFP64(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float64", low=-1.0, high=1.0),
            'y': self.random([32, 64], "float64", low=-1.0, high=1.0),
        }


class TestAtan2OpINT32(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", low=-10, high=10),
            'y': self.random([32, 64], "int32", low=-10, high=10),
        }


class TestAtan2OpINT64(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int64", low=-10, high=10),
            'y': self.random([32, 64], "int64", low=-10, high=10),
        }


class TestAtan2OpZeroCase1(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int64", low=0, high=1),
            'y': self.random([32, 64], "int64", low=0, high=1),
        }


class TestAtan2OpZeroCase2(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int64"),
            'y': self.random([32, 64], "int64", low=0, high=1),
        }


class TestAtan2OpZeroCase3(TestAtan2Op):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int64", low=0, high=1),
            'y': self.random([32, 64], "int64"),
        }


if __name__ == "__main__":
    unittest.main()
