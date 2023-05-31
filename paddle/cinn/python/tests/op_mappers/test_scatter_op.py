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

import random
import unittest

import numpy as np
from op_mapper_test import OpMapperTest

import paddle


class TestScatterOp(OpMapperTest):
    def init_input_data(self):
        dim0 = 6
        dim1 = 10
        x_data = self.random([dim0, dim1], "float32")
        ids_data = np.random.randint(
            0, dim0, [random.randint(1, 5)], dtype=np.int32
        )
        updates_data = self.random([len(ids_data), dim1], "float32")
        self.feed_data = {'x': x_data, 'ids': ids_data, 'updates': updates_data}

    def set_op_type(self):
        return "scatter"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        ids = paddle.static.data(
            name='ids',
            shape=self.feed_data['ids'].shape,
            dtype=self.feed_data['ids'].dtype,
        )
        updates = paddle.static.data(
            name='updates',
            shape=self.feed_data['updates'].shape,
            dtype=self.feed_data['updates'].dtype,
        )
        return {'X': [x], 'Ids': [ids], 'Updates': [updates]}

    def set_op_attrs(self):
        return {'overwrite': False}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestScatterOpOverWrite(TestScatterOp):
    def init_input_data(self):
        dim0 = 6
        dim1 = 10
        x_data = self.random([dim0, dim1], "float32")
        ids_data = np.random.randint(
            0, dim0, [random.randint(1, 10)], dtype=np.int32
        )
        # remove duplicate elements, because paddle has undetermined behavior for duplicate elements
        ids_data = np.unique(ids_data)
        updates_data = self.random([len(ids_data), dim1], "float32")
        self.feed_data = {'x': x_data, 'ids': ids_data, 'updates': updates_data}

    def set_op_attrs(self):
        return {'overwrite': True}


if __name__ == "__main__":
    unittest.main()
