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

import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestCbrtOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(
                self.case["shape"], self.case["dtype"], -100.0, 100.0
            ),
        }

    def build_paddle_program(self, target):
        numpy_out = np.cbrt(self.inputs["x"])
        out = paddle.to_tensor(numpy_out, stop_gradient=False)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("cbrt")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.cbrt(x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(
            max_relative_error=1e-3, max_absolute_error=1e-3
        )


class TestCbrtOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCbrtOpShape"
        self.cls = TestCbrtOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
            {
                "shape": [80, 1, 5, 7],
            },
            {
                "shape": [80, 3, 1024, 7],
            },
            {
                "shape": [10, 5, 1024, 2048],
            },
            {
                "shape": [1],
            },
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            {
                "shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestCbrtOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCbrtOpDtype"
        self.cls = TestCbrtOp
        self.inputs = [
            {
                "shape": [1],
            },
            {
                "shape": [5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestCbrtOpShape().run()
    TestCbrtOpDtype().run()
