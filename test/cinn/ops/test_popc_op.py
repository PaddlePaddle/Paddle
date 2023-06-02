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


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestPopcOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            # "x": self.random([32, 64], 'int32', low = -2147483648, high=2147483647)
            "x": np.array(
                [
                    -1591895863,
                    -1770335025,
                    -1290313501,
                    478042597,
                    189030958,
                    -935228100,
                    718518127,
                    -2066013593,
                    -1028229638,
                    -1930307001,
                    -858478166,
                    -282304333,
                ]
            ).astype(np.int32)
        }
        self.outputs = {
            "y": np.array(
                [14, 19, 16, 18, 12, 13, 20, 15, 19, 17, 16, 17]
            ).astype(np.int32)
        }

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("popc")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.popc(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestPopcCase1(TestPopcOp):
    def init_case(self):
        self.inputs = {
            # "x": self.random([48, 36], 'int32', low = -2147483648, high=2147483647)
            "x": np.array(
                [
                    [
                        -780762106,
                        2088944770,
                        1793870564,
                        995233974,
                        -1566864405,
                        -1550063384,
                    ],
                    [
                        58189437,
                        -585656506,
                        1058816786,
                        -1676158651,
                        -175192886,
                        2129254990,
                    ],
                ]
            ).astype(np.int32)
        }
        self.outputs = {
            "y": np.array(
                [[13, 12, 16, 14, 18, 17], [19, 18, 14, 16, 17, 20]]
            ).astype(np.int32)
        }


class TestPopcCase2(TestPopcOp):
    def init_case(self):
        self.inputs = {
            # "x": self.random([4, 3, 5, 8], 'int64', low = -9223372036854775808, high=9223372036854775807)
            "x": np.array(
                [
                    -2603587548323400654,
                    5370659515557365091,
                    -2051413160116828951,
                    9015154622229049624,
                    -8328245342679021727,
                    -8113334794330105534,
                    7187230222985732039,
                    1835610600500058242,
                ]
            ).astype(np.int64)
        }
        self.outputs = {
            "y": np.array([34, 32, 34, 32, 29, 31, 34, 32]).astype(np.int64)
        }


if __name__ == "__main__":
    unittest.main()
