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
class TestIsCloseOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([16, 16], "float32")}
        self.inputs['y'] = self.inputs["x"]
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        shape = paddle.broadcast_shape(x.shape, y.shape)
        x = paddle.broadcast_to(x, shape)
        y = paddle.broadcast_to(y, shape)

        out = paddle.isclose(x, y, self.rtol, self.atol, self.equal_nan)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("isclose")

        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = builder.isclose(x, y, self.rtol, self.atol, self.equal_nan)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestIsCloseOpCase1(TestIsCloseOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([16, 16], "float32"),
            "y": self.random([16, 16], "float32"),
        }
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False


class TestIsCloseOpCase2(TestIsCloseOp):
    def init_case(self):
        self.inputs = {
            "x": np.array([np.nan] * 32).astype("float32"),
            "y": self.random([32], "float32"),
        }
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False


class TestIsCloseOpCase3(TestIsCloseOp):
    def init_case(self):
        self.inputs = {
            "x": np.array([np.nan] * 32).astype("float32"),
            "y": np.array([np.nan] * 32).astype("float32"),
        }
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = True


class TestIsCloseOpCase4(TestIsCloseOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([16, 16], "float32"),
            "y": self.random([1], "float32"),
        }
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False


if __name__ == "__main__":
    unittest.main()
