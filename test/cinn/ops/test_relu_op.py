#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
import paddle.nn.functional as F


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestReluOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random(
                [
                    32,
                    64,
                ]
            ).astype("float32"),
            "dout": np.random.random((32, 64)).astype("float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = F.relu(x)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads(
            [out], [x], [self.inputs["dout"]]
        )

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("relu")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.relu(x)

        dout = builder.create_input(
            self.nptype2cinntype(self.inputs["dout"].dtype),
            self.inputs["dout"].shape,
            "dout",
        )
        x_grad = builder.relu_grad(dout, out)
        prog = builder.build()

        res = self.get_cinn_output(
            prog,
            target,
            [x, dout],
            [self.inputs["x"], self.inputs["dout"]],
            [out, x_grad],
            passes=[],
        )

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
