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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestMaxOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=self.case["x_low"],
            high=self.case["x_high"],
        )
        self.y_np = self.random(
            shape=self.case["y_shape"],
            dtype=self.case["y_dtype"],
            low=self.case["y_low"],
            high=self.case["y_high"],
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        y = paddle.to_tensor(self.y_np, stop_gradient=True)
        out = paddle.maximum(x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pow")
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
        out = builder.max(x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestMaxOpBase(TestCaseHelper):
    inputs = [
        {
            "x_shape": [1],
            "y_shape": [1],
        },
        {
            "x_shape": [32, 64],
            "y_shape": [32, 64],
        },
        {
            "x_shape": [2, 3, 4],
            "y_shape": [2, 3, 4],
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

    dtypes = [
        {
            "x_dtype": "float32",
            "y_dtype": "float32",
        },
    ]

    attrs = [
        {"x_low": -100, "x_high": 100, "y_low": -100, "y_high": 100},
    ]

    def init_attrs(self):
        self.class_name = "TestMaxOpBase"
        self.cls = TestMaxOp


class TestMaxOpShapeTest(TestMaxOpBase):
    def init_attrs(self):
        self.class_name = "TestMaxOpShapeTest"
        self.cls = TestMaxOp
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
                "x_shape": [2048],
                "y_shape": [2048],
            },
            {
                "x_shape": [32, 64],
                "y_shape": [32, 64],
            },
            {
                "x_shape": [2, 3, 4],
                "y_shape": [2, 3, 4],
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [16, 8, 4, 1024],
                "y_shape": [16, 8, 4, 1024],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "y_shape": [16, 8, 4, 2, 1],
            },
            {
                "x_shape": [1, 1, 1, 1, 1],
                "y_shape": [1, 1, 1, 1, 1],
            },
        ]


class TestMaxOpDtypeTest(TestMaxOpBase):
    def init_attrs(self):
        self.class_name = "TestMaxOpDtypeTest"
        self.cls = TestMaxOp
        self.dtypes = [
            # {
            # "x_dtype": "int8",
            # "y_dtype": "int8",
            # }, {
            # "x_dtype": "int16",
            # "y_dtype": "int16",
            # }, {
            # "x_dtype": "uint8",
            # "y_dtype": "uint8",
            # }, {
            # "x_dtype": "uint16",
            # "y_dtype": "uint16",
            # },
            {
                "x_dtype": "int32",
                "y_dtype": "int32",
            },
            {
                "x_dtype": "int64",
                "y_dtype": "int64",
            },
            # {
            #    "x_dtype": "float16",
            #    "y_dtype": "float16",
            #    "max_relative_error": 1e-3,
            # },
            {
                "x_dtype": "float32",
                "y_dtype": "float32",
            },
            {
                "x_dtype": "float64",
                "y_dtype": "float64",
            },
        ]


class TestMaxOpPolarityTest(TestMaxOpBase):
    def init_attrs(self):
        self.class_name = "TestMaxOpPolarityTest"
        self.cls = TestMaxOp
        self.attrs = [
            {
                "x_low": -100,
                "x_high": 100,
                "y_low": -100,
                "y_high": 100,
            }
        ]


class TestMaxOpBroadcastTest(TestMaxOpBase):
    def init_attrs(self):
        self.class_name = "TestMaxOpBroadcastTest"
        self.cls = TestMaxOp
        self.inputs = [
            {
                "x_shape": [32],
                "y_shape": [1],
            },
            {
                "x_shape": [1],
                "y_shape": [32],
            },
            {
                "x_shape": [1, 64],
                "y_shape": [32, 1],
            },
            {
                "x_shape": [1, 64],
                "y_shape": [32, 64],
            },
            {
                "x_shape": [32, 1],
                "y_shape": [32, 64],
            },
            {
                "x_shape": [1, 1],
                "y_shape": [32, 64],
            },
            {
                "x_shape": [1, 3, 4],
                "y_shape": [2, 3, 4],
            },
            {
                "x_shape": [1, 3, 1],
                "y_shape": [2, 3, 4],
            },
            {
                "x_shape": [1, 1, 1],
                "y_shape": [2, 3, 4],
            },
            {
                "x_shape": [2, 1, 1],
                "y_shape": [1, 3, 4],
            },
            {
                "x_shape": [1, 8, 4, 2],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [16, 8, 1, 1],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [1, 8, 1, 1],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [1, 8, 1, 2],
                "y_shape": [16, 1, 4, 1],
            },
            {
                "x_shape": [1, 8, 4, 2, 32],
                "y_shape": [16, 8, 4, 2, 32],
            },
            {
                "x_shape": [16, 1, 1, 2, 32],
                "y_shape": [16, 8, 4, 2, 32],
            },
            {
                "x_shape": [16, 1, 4, 1, 1],
                "y_shape": [16, 8, 4, 2, 32],
            },
            {
                "x_shape": [1, 1, 1, 1, 32],
                "y_shape": [16, 8, 4, 2, 32],
            },
            {
                "x_shape": [1, 1, 1, 1, 1],
                "y_shape": [16, 8, 4, 2, 32],
            },
            {
                "x_shape": [16, 1, 4, 1, 32],
                "y_shape": [1, 8, 1, 2, 1],
            },
        ]


if __name__ == "__main__":
    TestMaxOpShapeTest().run()
    TestMaxOpDtypeTest().run()
    TestMaxOpPolarityTest().run()
    TestMaxOpBroadcastTest().run()
