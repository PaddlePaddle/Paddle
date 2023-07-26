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


class TestTranspose2Op(OpMapperTest):
    def init_input_dtype(self):
        self.dtype = 'float32'

    def init_input_data(self):
        self.init_input_dtype()
        self.feed_data = {
            'x': self.random([2, 3, 4], self.dtype, 0.0, 100.0),
        }

    def set_op_type(self):
        return "transpose2"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"axis": [0, 2, 1]}

    def set_op_outputs(self):
        return {
            'Out': [str(self.feed_data['x'].dtype)],
            'XShape': [str(self.feed_data['x'].dtype)],
        }

    def skip_check_outputs(self):
        # in Paddle, XShape is None, its memory has been optimized
        return {"XShape"}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTranspose2OpInt32(TestTranspose2Op):
    def init_input_dtype(self):
        self.dtype = 'int32'


class TestTranspose2OpInt64(TestTranspose2Op):
    def init_input_dtype(self):
        self.dtype = 'int64'


if __name__ == "__main__":
    unittest.main()
