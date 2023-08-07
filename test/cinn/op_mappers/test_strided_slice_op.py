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


class TestStridedSliceOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([10], dtype='float32'),
        }
        self.axes = [0]
        self.starts = [2]
        self.ends = [5]
        self.strides = [1]
        self.infer_flags = [1]

    def set_op_type(self):
        return "strided_slice"

    def set_op_inputs(self):
        inputs = paddle.static.data(
            name='inputs',
            shape=self.feed_data['inputs'].shape,
            dtype=self.feed_data['inputs'].dtype,
        )
        return {'Input': [inputs]}

    def set_op_attrs(self):
        return {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['inputs'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestStridedSliceCase1(TestStridedSliceOp):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([10, 12], 'float32'),
        }
        self.axes = [0, 1]
        self.starts = [1, 2]
        self.ends = [6, 10]
        self.strides = [1, 2]
        self.infer_flags = [1, 1]


class TestStridedSliceCase2(TestStridedSliceOp):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([2, 10, 5], 'float32'),
        }
        self.axes = [0, 1, 2]
        self.starts = [1, 2, 3]
        self.ends = [6, 10, 5]
        self.strides = [1, 2, 1]
        self.infer_flags = [1, 1, 1]


class TestStridedSliceCase3(TestStridedSliceOp):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([2, 15, 10], 'int32'),
        }
        self.axes = [0, 1, 2]
        self.starts = [1, 10, 3]
        self.ends = [6, 2, 5]
        self.strides = [1, -2, 1]
        self.infer_flags = [1, 1, 1]


class TestStridedSliceCase4(TestStridedSliceOp):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([12, 14], 'float32'),
        }
        self.axes = [0, 1]
        self.starts = [1, -2]
        self.ends = [6, -10]
        self.strides = [2, -2]
        self.infer_flags = [1, -1]


class TestStridedSliceCase5(TestStridedSliceOp):
    def init_input_data(self):
        self.feed_data = {
            'inputs': self.random([120], 'float32'),
        }
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-120]
        self.strides = [-4]
        self.infer_flags = [-1]


if __name__ == "__main__":
    unittest.main()
