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
class TestBroadcastToOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.case["d_shape"])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.broadcast_to(
            x,
            out_shape=self.case["d_shape"],
            broadcast_axes=self.case["broadcast_axes"],
        )

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


class TestBroadcastToAllOne(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBroadcastToOpCase"
        self.cls = TestBroadcastToOp
        self.inputs = [
            {
                "x_shape": [1],
                "d_shape": [1, 2],
                "broadcast_axes": [1],
            },
            {
                "x_shape": [5, 3],
                "d_shape": [4, 5, 3],
                "broadcast_axes": [1, 2],
            },
            {
                "x_shape": [4, 5, 3],
                "d_shape": [6, 4, 5, 3],
                "broadcast_axes": [1, 2, 3],
            },
            {
                "x_shape": [5, 4, 3, 2],
                "d_shape": [6, 5, 4, 3, 2],
                "broadcast_axes": [1, 2, 3, 4],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "d_shape": [32, 16, 8, 4, 2, 1],
                "broadcast_axes": [1, 2, 3, 4, 5],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float32",
            },
        ]
        self.attrs = []


class TestBroadcastToAllTwo(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBroadcastToOpCase"
        self.cls = TestBroadcastToOp
        self.inputs = [
            {
                "x_shape": [5, 3],
                "d_shape": [4, 5, 3],
                "broadcast_axes": [1, 2],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "bool",
            },
            # {
            #    "x_dtype": "int8",
            # },
            {
                "x_dtype": "int32",
            },
            {
                "x_dtype": "int64",
            },
            {
                "x_dtype": "float16",
            },
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = []


class TestBroadcastToOpNoAxes(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.case["d_shape"])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.broadcast_to(x, out_shape=self.case["d_shape"])

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


class TestBroadcastToOpNoAxesAllOne(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBroadcastToOpNoAxesCase"
        self.cls = TestBroadcastToOpNoAxes
        self.inputs = [
            {
                "x_shape": [1],
                "d_shape": [1, 2],
            },
            {
                "x_shape": [6],
                "d_shape": [4, 5, 6],
            },
            {
                "x_shape": [1, 1, 1],
                "d_shape": [4, 5, 3],
            },
            {
                "x_shape": [1, 1, 3],
                "d_shape": [4, 5, 3],
            },
            {
                "x_shape": [4, 1, 3],
                "d_shape": [4, 5, 3],
            },
            {
                "x_shape": [64, 2],
                "d_shape": [64, 2],
            },
            {
                "x_shape": [64, 32, 16],
                "d_shape": [128, 64, 32, 16],
            },
            {
                "x_shape": [64, 32, 16, 8],
                "d_shape": [128, 64, 32, 16, 8],
            },
            # {
            #    "x_shape": [128, 64, 32, 16, 8],
            #    "d_shape": [256, 128, 64, 32, 16, 8],
            # },
        ]
        self.dtypes = [
            {
                "x_dtype": "float32",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestBroadcastToAllOne().run()
    TestBroadcastToAllTwo().run()
    TestBroadcastToOpNoAxesAllOne().run()
