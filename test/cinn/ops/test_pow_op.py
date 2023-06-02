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

from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestPowOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([32, 64], "float32"),
            "y": self.random([32, 64], "float32", 0.0, 4.0),
        }
        self.axis = -1

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        out = paddle.pow(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pow")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        out = builder.pow(x, y, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestPowCase1(TestPowOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([8, 16, 32, 32], "float32"),
            "y": self.random([1], "float32", 0.0, 4.0),
        }
        self.axis = 0


class TestPowCase2(TestPowOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([8, 16, 32, 32], "int32", 2, 10),
            "y": self.random([8, 16, 32, 32], "int32", 0, 5),
        }
        self.axis = -1


class TestPowFP64(TestPowOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([8, 16, 32, 32], "float64", 2, 10),
            "y": self.random([8, 16, 32, 32], "float64", 0, 5),
        }
        self.axis = -1


if __name__ == "__main__":
    unittest.main()
