#!/usr/bin/env python3

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

import numpy as np
from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestTopKOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.k = 1
        self.axis = 1

    def build_paddle_program(self, target):
        axis = -1
        x1 = paddle.to_tensor(self.inputs["x1"], stop_gradient=True)
        out = paddle.topk(x1, self.k, self.axis)

        self.paddle_outputs = [out[0], out[1]]

    def build_cinn_program(self, target):
        builder = NetBuilder("sum")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        out = builder.top_k(x1, self.k, self.axis, False)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x1], [self.inputs["x1"]], [out[0], out[1]]
        )

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTopKCase1(TestTopKOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.k = 2
        self.axis = 1


class TestTopKCase2(TestTopKOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.k = 1
        self.axis = 0


if __name__ == "__main__":
    unittest.main()
