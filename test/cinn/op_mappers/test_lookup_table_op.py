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


class TestLookupTableOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            "w": self.random([10, 3], "float32"),
            "ids": self.random([5, 1], "int64", 0, 9),
        }

    def set_op_type(self):
        return "lookup_table"

    def set_op_inputs(self):
        w = paddle.static.data(
            name="w",
            shape=self.feed_data["w"].shape,
            dtype=self.feed_data["w"].dtype,
        )
        ids = paddle.static.data(
            name="ids",
            shape=self.feed_data["ids"].shape,
            dtype=self.feed_data["ids"].dtype,
        )
        return {"W": [w], "Ids": [ids]}

    def set_op_attrs(self):
        return {"padding_idx": -1}

    def set_op_outputs(self):
        return {"Out": [str(self.feed_data["w"].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestLookupTableOpCase1(TestLookupTableOp):
    def init_input_data(self):
        self.feed_data = {
            "w": self.random([32, 64], "float64"),
            "ids": self.random([10, 1], "int64", 0, 31),
        }

    def set_op_attrs(self):
        return {"padding_idx": 1}


class TestLookupTableV2Op(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            "w": self.random([10, 3], "float32"),
            "ids": self.random([5, 2], "int32", 0, 9),
        }

    def set_op_type(self):
        return "lookup_table_v2"

    def set_op_inputs(self):
        w = paddle.static.data(
            name="w",
            shape=self.feed_data["w"].shape,
            dtype=self.feed_data["w"].dtype,
        )
        ids = paddle.static.data(
            name="ids",
            shape=self.feed_data["ids"].shape,
            dtype=self.feed_data["ids"].dtype,
        )
        return {"W": [w], "Ids": [ids]}

    def set_op_attrs(self):
        return {"padding_idx": -1}

    def set_op_outputs(self):
        return {"Out": [str(self.feed_data["w"].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestLookupTableV2OpCase1(TestLookupTableV2Op):
    def init_input_data(self):
        self.feed_data = {
            "w": self.random([32, 64], "float64"),
            "ids": self.random([10, 3], "int64", 0, 31),
        }

    def set_op_attrs(self):
        return {"padding_idx": 1}


if __name__ == "__main__":
    unittest.main()
