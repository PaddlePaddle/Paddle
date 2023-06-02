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
class TestLookupTableOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "table": np.random.random(
                [
                    10,
                    20,
                ]
            ).astype("float32"),
            "ids": np.random.random_integers(0, 9, (5, 2)).astype("int64"),
        }

    def build_paddle_program(self, target):
        table = paddle.to_tensor(self.inputs["table"], stop_gradient=False)
        ids = paddle.to_tensor(self.inputs["ids"], stop_gradient=False)
        out = F.embedding(ids, table, 1)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("lookup_table")
        table = builder.create_input(
            Float(32), self.inputs["table"].shape, "table"
        )
        ids = builder.create_input(
            Int(64), self.inputs["ids"].shape + (1,), "ids"
        )
        out = builder.lookup_table(table, ids, 1)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog,
            target,
            [table, ids],
            [self.inputs["table"], self.inputs["ids"]],
            [out],
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
