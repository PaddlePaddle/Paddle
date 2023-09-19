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
class TestSubOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if self.case["broadcast"]:
            self.inputs = {
                "x": self.random(self.case["x_shape"], self.case["dtype"]),
                "y": self.random(self.case["y_shape"], self.case["dtype"]),
            }
        else:
            self.inputs = {
                "x": self.random(self.case["shape"], self.case["dtype"]),
                "y": self.random(self.case["shape"], self.case["dtype"]),
            }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        out = paddle.subtract(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sub")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        out = builder.subtract(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSubOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSubOpShapeTest"
        self.cls = TestSubOp
        self.inputs = [
            {
                "shape": [64],
            },
            {
                "shape": [64, 32],
            },
            {
                "shape": [64, 1],
            },
            {
                "shape": [64, 32, 128],
            },
            {
                "shape": [1, 32, 128],
            },
            {
                "shape": [64, 32, 16, 32],
            },
            {
                "shape": [64, 32, 1, 32],
            },
            {
                "shape": [64, 32, 16, 1, 128],
            },
            {
                "shape": [1],
            },
            {
                "shape": [1, 1],
            },
            {
                "shape": [1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1024, 1, 1],
            },
            {
                "shape": [65536],
            },
            {
                "shape": [131072],
            },
            {"shape": [1048576]},
            {
                "shape": [64, 32, 16, 8, 4],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"broadcast": False}]


class TestSubOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSubOpDtypeTest"
        self.cls = TestSubOp
        self.inputs = [
            {
                "shape": [64, 1, 128],
            },
            {
                "shape": [64, 32, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [{"axes": []}]
        self.attrs = [{"broadcast": False}]


class TestSubOpBroadcastTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSubOpShapeTest"
        self.cls = TestSubOp
        self.inputs = [
            {
                "x_shape": [64],
                "y_shape": [1],
            },
            {
                "x_shape": [1],
                "y_shape": [64],
            },
            {
                "x_shape": [64, 32],
                "y_shape": [64, 1],
            },
            {
                "x_shape": [1, 1],
                "y_shape": [64, 32],
            },
            {
                "x_shape": [64, 1],
                "y_shape": [1, 32],
            },
            {
                "x_shape": [64, 1, 128],
                "y_shape": [64, 32, 128],
            },
            {
                "x_shape": [64, 32, 128],
                "y_shape": [64, 32, 1],
            },
            {
                "x_shape": [64, 1, 128],
                "y_shape": [1, 32, 128],
            },
            {
                "x_shape": [1, 1, 1],
                "y_shape": [64, 32, 128],
            },
            {
                "x_shape": [64, 1, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 32],
                "y_shape": [64, 32, 1, 32],
            },
            {
                "x_shape": [64, 1, 1, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 1],
                "y_shape": [64, 1, 16, 32],
            },
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [1, 32, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [65536],
                "y_shape": [1],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"broadcast": True}]


if __name__ == "__main__":
    TestSubOpShapeTest().run()
    TestSubOpDtypeTest().run()
    TestSubOpBroadcastTest().run()
