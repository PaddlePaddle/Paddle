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
from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestMaxOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random((16, 64)).astype("float32"),
            "y": np.random.random((16, 64)).astype("float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        out = paddle.maximum(x, y)

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
        out = builder.max(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestMinOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random((16, 64)).astype("float32"),
            "y": np.random.random((16, 64)).astype("float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        out = paddle.minimum(x, y)

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
        out = builder.min(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
