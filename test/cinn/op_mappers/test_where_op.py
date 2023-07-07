#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANy KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from op_mapper_test import OpMapperTest

import paddle


class TestWhereOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'condition': self.random([2, 3], 'bool'),
            'x': self.random([2, 3], 'float32'),
            'y': self.random([2, 3], 'float32'),
        }

    def set_op_type(self):
        return "where"

    def set_op_inputs(self):
        condition = paddle.static.data(
            name='condition',
            shape=self.feed_data['condition'].shape,
            dtype=self.feed_data['condition'].dtype,
        )
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
        return {'Condition': [condition], 'X': [x], "Y": [y]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
