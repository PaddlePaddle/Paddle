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


def infer_broadcast_shape(arr, indices, axis):
    # This function is used in take/put_along_axis
    broadcast_shape_list = list(arr.shape)
    broadcast_shape_list[axis] = list(indices.shape)[axis]
    broadcast_shape = tuple(broadcast_shape_list)
    for i in range(len(arr.shape)):
        if arr.shape[i] < indices.shape[i]:
            # if indices matrix has larger size than arr matrix, do not broadcast.
            return None
    return broadcast_shape


class TestTakeAlongAxisOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 1, 5], 'int32', 0, 5),
        }
        self.axis = 0

    def set_op_type(self):
        return "take_along_axis"

    def set_op_inputs(self):
        broadcast_shape = infer_broadcast_shape(
            self.feed_data['x'], self.feed_data['index'], self.axis
        )
        if not broadcast_shape:
            broadcast_shape = self.feed_data['index'].shape
        self.feed_data['index'] = np.broadcast_to(
            self.feed_data['index'], broadcast_shape
        ).copy()
        broadcast_shape_list = list(broadcast_shape)
        broadcast_shape_list[self.axis] = list(self.feed_data['x'].shape)[
            self.axis
        ]
        broadcast_shape = tuple(broadcast_shape_list)
        self.feed_data['x'] = np.broadcast_to(
            self.feed_data['x'], broadcast_shape
        ).copy()
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        index = paddle.static.data(
            name='index',
            shape=self.feed_data['index'].shape,
            dtype=self.feed_data['index'].dtype,
        )
        return {'Input': [x], 'Index': [index]}

    def set_op_attrs(self):
        return {'Axis': self.axis}

    def set_op_outputs(self):
        return {'Result': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTakeAlongAxisCase1(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 1, 5], 'int32', 0, 5),
        }
        self.axis = 1


class TestTakeAlongAxisCase2(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 1, 5], 'int32', 0, 5),
        }
        self.axis = 2


class TestTakeAlongAxisCase3(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 1, 5], 'int32', 0, 5),
        }
        self.axis = 3


class TestTakeAlongAxisCase4(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([5, 1, 1, 1], 'int32', 0, 5),
        }
        self.axis = 0


class TestTakeAlongAxisCase5(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 5, 1, 1], 'int32', 0, 5),
        }
        self.axis = 0


class TestTakeAlongAxisCase6(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 5, 1], 'int32', 0, 5),
        }
        self.axis = 0


class TestTakeAlongAxisCase7(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float32'),
            'index': self.random([1, 1, 1, 5], 'int64', 0, 5),
        }
        self.axis = 0


class TestTakeAlongAxisCase8(TestTakeAlongAxisOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([5, 5, 5, 5], 'float64'),
            'index': self.random([1, 1, 1, 5], 'int64', 0, 5),
        }
        self.axis = 0


if __name__ == "__main__":
    unittest.main()
