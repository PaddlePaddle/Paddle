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


class TestScaleOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.scale = -1.0
        self.bias = 0.0
        self.bias_after_scale = True

    def set_op_type(self):
        return "scale"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "scale": self.scale,
            "bias": self.bias,
            "bias_after_scale": self.bias_after_scale,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestScaleCase1(TestScaleOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.scale = 2.0
        self.bias = 1.0
        self.bias_after_scale = True


class TestScaleCase2(TestScaleOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}
        self.scale = 2.0
        self.bias = 1.0
        self.bias_after_scale = False


class TestScaleWithScaleTensor(TestScaleOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
            "scale": self.random([1], "float32", 2.0, 10.0),
        }
        self.bias = 2.0
        self.bias_after_scale = True

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
        return {'X': [x], "ScaleTensor": [scale]}

    def set_op_attrs(self):
        return {"bias": self.bias, "bias_after_scale": self.bias_after_scale}


class TestScaleWithScaleTensorCase1(TestScaleWithScaleTensor):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
            "scale": self.random([1], "float32", 2.0, 10.0),
        }
        self.bias = 0.0
        self.bias_after_scale = True


class TestScaleWithScaleTensorCase2(TestScaleWithScaleTensor):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32"),
            "scale": self.random([1], "float32", 2.0, 10.0),
        }
        self.bias = 0.0
        self.bias_after_scale = True


if __name__ == "__main__":
    unittest.main()
