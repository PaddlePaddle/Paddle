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

import numpy as np
from op_mapper_test import OpMapperTest

import paddle


class TestTileOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.repeat_times = [2, 2]

    def set_op_type(self):
        return "tile"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"repeat_times": self.repeat_times}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTileCase1(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], 'float32'),
        }
        self.repeat_times = [1, 2, 3]


class TestTileCase2(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 10, 5], 'float32'),
        }
        self.repeat_times = [2, 2]


class TestTileCase3(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 4, 15], 'float32'),
        }
        self.repeat_times = [2, 1, 4]


class TestTileCase4(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([12, 14], 'float32'),
        }
        self.repeat_times = [2, 3]


class TestTileCase5(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([120], 'float32'),
        }
        self.repeat_times = [2, 2]


if __name__ == "__main__":
    unittest.main()
