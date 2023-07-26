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
class TestCastOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.cast(x, self.case["d_dtype"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("cast")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.cast(x, self.case["d_dtype"])

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


class TestCastShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCastOpCase"
        self.cls = TestCastOp
        self.inputs = [
            {
                "x_shape": [1],
            },
            {
                "x_shape": [1024],
            },
            {
                "x_shape": [32, 64],
            },
            {
                "x_shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float32",
            }
        ]
        self.attrs = [
            {
                "d_dtype": "float64",
            }
        ]


class TestCastDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCastOpCase"
        self.cls = TestCastOp
        self.inputs = [
            {
                "x_shape": [32, 64],
            }
        ]
        self.dtypes = [
            {
                "x_dtype": "bool",
            },
            {
                "x_dtype": "int8",
            },
            {"x_dtype": "int16"},
            {
                "x_dtype": "int32",
            },
            {"x_dtype": "int64"},
            {"x_dtype": "float16", "max_relative_error": 1e-3},
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = [
            {
                "d_dtype": "bool",
            },
            {
                "d_dtype": "int8",
            },
            {"d_dtype": "int16"},
            {
                "d_dtype": "int32",
            },
            {"d_dtype": "int64"},
            {"d_dtype": "float16", "max_relative_error": 1e-3},
            {
                "d_dtype": "float32",
            },
            {
                "d_dtype": "float64",
            },
        ]


if __name__ == "__main__":
    TestCastShape().run()
    TestCastDtype().run()
