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

from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestTransposeOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([2, 3], "float32")}
        self.axes = [1, 0]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.transpose(
            x,
            [
                axis + len(self.inputs["x"].shape) if axis < 0 else axis
                for axis in self.axes
            ],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("transpose_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs['x'].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.transpose(x, self.axes)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTransposeOpWithNegAxis(TestTransposeOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 8, 2], "float32")}
        self.axes = [-1, 1, -3]


if __name__ == "__main__":
    unittest.main()
