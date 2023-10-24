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


class TestLayerNormOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4, 5], 'float32'),
            'scale': self.random([60], 'float32', 1.0, 2.0),
            'bias': self.random([60], 'float32', -10.0, 10.0),
        }
        self.beigin_norm_axis = 1

    def set_op_type(self):
        return "layer_norm"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        scale = paddle.static.data(
            name='scale',
            shape=self.feed_data['scale'].shape,
            dtype=self.feed_data['scale'].dtype,
        )
        bias = paddle.static.data(
            name='bias',
            shape=self.feed_data['bias'].shape,
            dtype=self.feed_data['bias'].dtype,
        )
        return {'X': [x], 'Scale': [scale], "Bias": [bias]}

    def set_op_attrs(self):
        return {"epsilon": 1e-5, "begin_norm_axis": self.beigin_norm_axis}

    def set_op_outputs(self):
        return {
            'Y': [str(self.feed_data['x'].dtype)],
            'Mean': [str(self.feed_data['scale'].dtype)],
            'Variance': [str(self.feed_data['scale'].dtype)],
        }

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestLayerNormFp16(TestLayerNormOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4, 5], 'float16'),
            'scale': self.random([60], 'float32', 1.0, 2.0),
            'bias': self.random([60], 'float32', -10.0, 10.0),
        }
        self.beigin_norm_axis = 1

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


if __name__ == "__main__":
    unittest.main()
