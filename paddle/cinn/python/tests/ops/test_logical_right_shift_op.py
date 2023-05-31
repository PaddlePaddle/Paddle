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
class TestLogicalRightShift(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            # "x": self.random([1, 24], 'int32', low = -2147483648, high=2147483647)
            "x": np.array(
                [
                    [
                        1690476611,
                        142184466,
                        -1752569340,
                        1860589058,
                        -1295695292,
                        1912939056,
                        -1416770533,
                        -483282486,
                        284237925,
                        -2094465968,
                        -823026780,
                        -1503970769,
                        -535860601,
                        1515033359,
                        -1212100470,
                        -2008734407,
                        704803066,
                        1861454881,
                        -479224831,
                        1939718614,
                        -1903975007,
                        -1197706543,
                        1327016838,
                        -232019105,
                    ]
                ]
            ).astype(np.int32),
            # "y": self.random([1, 24], 'int32', low = 0, high=32)
            "y": np.array(
                [
                    [
                        20,
                        3,
                        12,
                        3,
                        0,
                        31,
                        0,
                        2,
                        6,
                        16,
                        1,
                        7,
                        6,
                        2,
                        19,
                        16,
                        7,
                        17,
                        10,
                        15,
                        8,
                        9,
                        24,
                        4,
                    ]
                ]
            ).astype(np.int32),
        }
        self.outputs = {
            "out": np.array(
                [
                    [
                        1612,
                        17773058,
                        620702,
                        232573632,
                        -1295695292,
                        0,
                        -1416770533,
                        952921202,
                        4441217,
                        33576,
                        1735970258,
                        21804660,
                        58736042,
                        378758339,
                        5880,
                        34885,
                        5506273,
                        14201,
                        3726311,
                        59195,
                        9339813,
                        6049337,
                        79,
                        253934261,
                    ]
                ]
            ).astype(np.int32)
        }

    def build_paddle_program(self, target):
        out = paddle.to_tensor(self.outputs["out"], stop_gradient=False)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("logical_right_shift")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        out = builder.logical_right_shift(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
