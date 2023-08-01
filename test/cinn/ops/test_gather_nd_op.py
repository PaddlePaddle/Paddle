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

import logging
import os
import unittest
from itertools import product

import numpy as np
from cinn.common import Int, is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="gather_nd")


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestGatherNdOp(OpTest):
    def setUp(self):
        self.data = []
        self.init_case()

    def init_case(self):
        self.inputs = [{"x": [3, 4, 3], "index": [4, 1]}]
        self.dtypes = ["float32"]

    def build_paddle_program(self, target):
        for inputs, dtype in product(self.inputs, self.dtypes):
            x_shape = inputs["x"]
            index_shape = inputs["index"]
            x = np.random.randn(*x_shape).astype(dtype)
            index = np.random.randint(0, min(x_shape), index_shape).astype(
                "int32"
            )
            self.data.append([x, index])
            x = paddle.to_tensor(x, stop_gradient=False)
            index = paddle.to_tensor(index, stop_gradient=False)
            out = paddle.gather_nd(x, index)
            logger.debug(f" -- The output of Paddle:\n{out}")
            self.paddle_outputs.append(out)

    def build_cinn_program(self, target):
        for i, (inputs, dtype) in enumerate(product(self.inputs, self.dtypes)):
            builder = NetBuilder("gather")
            x = builder.create_input(
                self.nptype2cinntype(dtype), inputs["x"], "x"
            )
            index = builder.create_input(Int(32), inputs["index"], "index")
            out = builder.gather_nd(x, index)
            prog = builder.build()
            res = self.get_cinn_output(
                prog, target, [x, index], self.data[i], [out]
            )
            logger.debug(f" -- The output of CINN:\n{res}")
            self.cinn_outputs.extend(res)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestGatherOpAll(TestGatherNdOp):
    def init_case(self):
        self.inputs = []
        for x_shape in [
            [16],
            [8, 16],
            [4, 8, 16],
            [2, 4, 8, 16],
            [2, 4, 8, 1],
            [2, 4, 8, 1024],
        ]:
            for j in range(1, len(x_shape)):
                self.inputs.append({"x": x_shape, "index": [8, j]})

        self.dtypes = [
            "float32",
            "float64",
            "int16",
            "int32",
            "int64",
            # "uint8"  # note: some types is not supported in paddle now.
        ]


if __name__ == "__main__":
    unittest.main()
