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


import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestLeftShiftOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=-100,
            high=100,
        )
        self.y_np = self.random(
            shape=self.case["y_shape"],
            dtype=self.case["y_dtype"],
            low=0,
            high=16,
        )

    def build_paddle_program(self, target):
        np_out = np.left_shift(self.x_np, self.y_np)
        out = paddle.to_tensor(np_out, stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("left_shift")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.case["y_dtype"]),
            self.case["y_shape"],
            "y",
        )
        out = builder.left_shift(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out]
        )
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestLeftShiftAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLeftShiftOpCase"
        self.cls = TestLeftShiftOp
        self.inputs = [
            {
                "x_shape": [1],
                "y_shape": [1],
            },
            {
                "x_shape": [1024],
                "y_shape": [1024],
            },
            {
                "x_shape": [512, 256],
                "y_shape": [512, 256],
            },
            {
                "x_shape": [128, 64, 32],
                "y_shape": [128, 64, 32],
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "y_shape": [16, 8, 4, 2, 1],
            },
        ]
        self.dtypes = [
            # {
            #     "x_dtype": "uint8",
            #     "y_dtype": "uint8",
            # },
            {
                "x_dtype": "int32",
                "y_dtype": "int32",
            },
            # {
            #     "x_dtype": "int64",
            #     "y_dtype": "int64",
            # },
        ]
        self.attrs = []


class TestLeftShiftAllWithBroadcast(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLeftShiftOpCase"
        self.cls = TestLeftShiftOp
        self.inputs = [
            {
                "x_shape": [1],
                "y_shape": [1],
            },
            {
                "x_shape": [1024],
                "y_shape": [1],
            },
            {
                "x_shape": [512, 256],
                "y_shape": [1, 1],
            },
            {
                "x_shape": [128, 64, 32],
                "y_shape": [1, 1, 1],
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [1, 1, 1, 1],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "y_shape": [1, 1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            # {
            #     "x_dtype": "uint8",
            #     "y_dtype": "uint8",
            # },
            {
                "x_dtype": "int32",
                "y_dtype": "int32",
            },
            # {
            #     "x_dtype": "int64",
            #     "y_dtype": "int64",
            # }
        ]
        self.attrs = []


if __name__ == "__main__":
    TestLeftShiftAll().run()
    TestLeftShiftAllWithBroadcast().run()
