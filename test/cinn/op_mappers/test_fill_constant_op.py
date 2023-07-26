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

import numpy as np
from op_mapper_test import OpMapperTest

import paddle


class TestFillConstantOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {"x": self.random([1], "float32")}
        self.shape = [10, 10]
        self.value = np.random.default_rng(12345).random()
        self.str_value = ""
        self.dtype = "float32"

    def set_op_type(self):
        return "fill_constant"

    def set_op_inputs(self):
        return {}

    def set_op_attrs(self):
        return {
            "shape": self.shape,
            "value": float(self.value),
            "str_value": self.str_value,
            "dtype": self.nptype2paddledtype(self.dtype),
        }

    def set_op_outputs(self):
        return {'Out': [self.dtype]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestFillConstantCase1(TestFillConstantOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [10, 10]
        self.value = np.random.default_rng(12345).integers(low=0, high=10000)
        self.str_value = ""
        self.dtype = "int32"


class TestFillConstantCase2(TestFillConstantOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [10, 10]
        self.value = 0
        self.str_value = "0.123456"
        self.dtype = "float32"


class TestFillConstantByValueTensor(TestFillConstantOp):
    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {"ValueTensor": [x]}


class TestFillConstantByValueTensorCase1(TestFillConstantByValueTensor):
    def init_input_data(self):
        self.feed_data = {"x": self.random([1], "int32", -10, 10)}
        self.shape = [10, 10]
        self.value = np.random.default_rng(12345).random()
        self.str_value = ""
        self.dtype = "float32"


if __name__ == "__main__":
    unittest.main()
