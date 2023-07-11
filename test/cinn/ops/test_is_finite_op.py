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


import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestIsFiniteOp(OpTest):
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

        index = np.random.randint(0, len(self.x_np))
        inf_data = np.where(self.x_np[index] > 0, np.inf, np.nan)
        self.x_np[index] = inf_data.astype(self.case["x_dtype"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.isfinite(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("is_finite")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.is_finite(x)

        prog = builder.build()

        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsFiniteOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsFiniteOpShape"
        self.cls = TestIsFiniteOp
        self.inputs = [
            {
                "x_shape": [1],
            },
            {
                "x_shape": [1024],
            },
            {
                "x_shape": [1, 2048],
            },
            {
                "x_shape": [1, 1, 1],
            },
            {
                "x_shape": [32, 64],
            },
            {
                "x_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float32",
            }
        ]
        self.attrs = []


class TestIsFiniteOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsFiniteOpDtype"
        self.cls = TestIsFiniteOp
        self.inputs = [
            {
                "x_shape": [32, 64],
            }
        ]
        self.dtypes = [
            {
                "x_dtype": "int32",
            },
            {
                "x_dtype": "int64",
            },
            {"x_dtype": "float16", "max_relative_error": 1e-3},
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestIsFiniteOpShape().run()
    TestIsFiniteOpDtype().run()
