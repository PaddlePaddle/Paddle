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


class TestConv2dOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 16, 32, 32], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32"),
        }
        self.data_format = 'NCHW'

    def set_op_type(self):
        return "conv2d"

    def set_op_inputs(self):
        x = paddle.static.data(
            'x', self.feed_data["x"].shape, self.feed_data["x"].dtype
        )
        weight = paddle.static.data(
            'weight',
            self.feed_data["weight"].shape,
            self.feed_data["weight"].dtype,
        )
        return {'Input': [x], 'Filter': [weight]}

    def set_op_attrs(self):
        return {
            "strides": [1, 1],
            "paddings": [0, 0],
            "dilations": [1, 1],
            "groups": 1,
            "data_format": self.data_format,
            "padding_algorithm": "EXPLICIT",
            "use_cudnn": True,
        }

    def set_op_outputs(self):
        return {'Output': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConv2dNCHWFP16(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 16, 32, 32], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16"),
        }
        self.data_format = 'NCHW'

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


class TestConv2dNHWC(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 32, 32, 16], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32"),
        }
        self.data_format = 'NHWC'


class TestConv2dNHWCFP16(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 32, 32, 16], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16"),
        }
        self.data_format = 'NHWC'

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


if __name__ == "__main__":
    unittest.main()
