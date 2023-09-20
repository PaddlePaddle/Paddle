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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestExpandDimsOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.init_case()

    def init_case(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.unsqueeze(x, self.case["axes_shape"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("expand_dims")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.expand_dims(x, self.case["axes_shape"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestExpandDimsAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestExpandDimsOpCase"
        self.cls = TestExpandDimsOp
        self.inputs = [
            {
                "x_shape": [1],
                "axes_shape": [0],
            },
            {
                "x_shape": [1024],
                "axes_shape": [0, 1],
            },
            {
                "x_shape": [32, 64],
                "axes_shape": [0, 2],
            },
            {
                "x_shape": [32, 64],
                "axes_shape": [0, 1, 2],
            },
            {
                "x_shape": [32, 64, 128],
                "axes_shape": [0, 1, 2],
            },
            {
                "x_shape": [32, 64, 128],
                "axes_shape": [1, 2, 3],
            },
            {
                "x_shape": [128, 64, 32, 16],
                "axes_shape": [0, 1],
            },
            {
                "x_shape": [128, 64, 32, 16],
                "axes_shape": [3, 4],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "axes_shape": [2],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "axes_shape": [5],
            },
        ]
        self.dtypes = [
            # {
            #    "x_dtype": "bool",
            #    "axes_dtype": "int32",
            # },
            # {
            #    "x_dtype": "int8",
            #    "axes_dtype": "int32",
            # },
            # {
            #    "x_dtype": "int16",
            #    "axes_dtype": "int32",
            # },
            # {
            #    "x_dtype": "int32",
            #    "axes_dtype": "int32",
            # },
            # {
            #    "x_dtype": "int64",
            #    "axes_dtype": "int32",
            # },
            # {
            #    "x_dtype": "float16",
            #    "max_relative_error": 1e-3,
            #    "axes_dtype": "int32",
            # },
            {
                "x_dtype": "float32",
            },
            # {
            #    "x_dtype": "float64",
            #    "axes_dtype": "int32",
            # },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestExpandDimsAll().run()
