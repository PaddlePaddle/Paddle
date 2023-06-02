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
class TestArgSortOp(OpTest):
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
        self.axis = 1
        self.descending = False

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x1"], stop_gradient=True)
        out = paddle.argsort(x1, self.axis, self.descending)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argsort")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        out = builder.argsort(x1, self.axis, not self.descending)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x1], [self.inputs["x1"]], out
        )

        self.cinn_outputs = np.array([forward_res[0]]).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgSortCase1(TestArgSortOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.axis = 0
        self.descending = False


class TestArgSortCase2(TestArgSortOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.axis = 0
        self.descending = True


class TestArgSortCase3(TestArgSortOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.axis = 1
        self.descending = True


class TestArgSortCase4(TestArgSortOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random(
                [
                    2,
                    4,
                ]
            ).astype("float32")
        }
        self.axis = -1
        self.descending = True


if __name__ == "__main__":
    unittest.main()
