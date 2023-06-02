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
import paddle.nn.functional as F


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestOneHotOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "X": np.random.random_integers(0, 9, (10)).astype("int64")
        }
        self.depth = 10
        self.axis = -1
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["X"])
        out = F.one_hot(x, self.depth)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("one_hot")
        x = builder.create_input(Int(64), self.inputs["X"].shape, "X")
        on_value = builder.fill_constant([1], 1, 'on_value', 'int64')
        off_value = builder.fill_constant([1], 0, 'off_value', 'int64')

        out = builder.one_hot(
            x, on_value, off_value, self.depth, self.axis, self.dtype
        )
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["X"]], [out]
        )

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.build_paddle_program(self.target)
        self.build_cinn_program(self.target)
        self.check_results(
            self.paddle_outputs, self.cinn_outputs, 1e-5, False, False
        )


if __name__ == "__main__":
    unittest.main()
