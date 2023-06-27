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
from op_test import OpTest, OpTestTool
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestElementwiseMulOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([32, 64]).astype("float32"),
            "y": np.random.random([32, 64]).astype("float32")
        }
        self.axis = 0

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.")
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = np.arange(0, axis).tolist() + np.arange(
                axis + y_rank, x_rank).tolist()

            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(self.inputs["x"].shape), len(self.inputs["y"].shape),
            self.axis)
        y_t = paddle.unsqueeze(
            y, axis=unsqueeze_axis) if len(unsqueeze_axis) > 0 else y
        out = paddle.multiply(x, y_t)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("multiply")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = builder.multiply(x, y, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y],
                                   [self.inputs["x"], self.inputs["y"]], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestMulCase1(TestElementwiseMulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 32, 32]).astype("float32"),
            "y": np.random.random([32, 32]).astype("float32")
        }
        self.axis = 2


if __name__ == "__main__":
    unittest.main()
