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


class TestAssignValueOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([10], 'float32')}

    def set_op_type(self):
        return "assign_value"

    def set_op_inputs(self):
        return {}

    def convert_values(self, dtype, values):
        if str(dtype) == "float64":
            return 'fp32_values', [float(v) for v in values]
        elif str(dtype) == "float32":
            return 'fp32_values', [float(v) for v in values]
        elif str(dtype) == "int64":
            return 'int64_values', [int(v) for v in values]
        elif str(dtype) == "int32":
            return 'int32_values', [int(v) for v in values]
        elif str(dtype) == "bool":
            return 'bool_values', [int(v) for v in values]
        else:
            self.assertTrue(
                False,
                msg="The data type of 'input' must be bool, float32, int32 or int64",
            )

    def set_op_attrs(self):
        dtype = self.feed_data['x'].dtype
        shape = self.feed_data['x'].shape
        value_name, values = self.convert_values(dtype, self.feed_data['x'])
        return {
            'dtype': self.nptype2paddledtype(
                str(dtype)
            ),  # should keep the same as input
            'shape': shape,
            value_name: values,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestAssignValueCase1(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([10], 'int32', 0, 1000)}


class TestAssignValueCase2(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([10], 'bool')}


class TestAssignValueCase3(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([10], 'int64', 0, 1000)}


class TestAssignValueCase4(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([1], 'float32')}


class TestAssignValueCase5(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([np.random.randint(100, 1000)], 'float32')
        }


class TestAssignValueCase6(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': np.arange(128, dtype="int64")}


class TestAssignValueCase7(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': np.arange(128, dtype="int32")}


class TestAssignValueCase8(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': np.arange(0.0, 12.8, 0.1, dtype="float32")}


class TestAssignValueCase9(TestAssignValueOp):
    def init_input_data(self):
        self.feed_data = {'x': np.arange(127, -1, -1, dtype="int32")}


if __name__ == "__main__":
    unittest.main()
