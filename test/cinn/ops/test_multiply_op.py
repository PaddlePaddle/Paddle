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
class TestElementwiseMulOp(OpTest):
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
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.",
            )
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = (
                np.arange(0, axis).tolist()
                + np.arange(axis + y_rank, x_rank).tolist()
            )
            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(x.shape), len(y.shape), self.case["axis"]
        )
        y_t = (
            paddle.unsqueeze(y, axis=unsqueeze_axis)
            if len(unsqueeze_axis) > 0
            else y
        )
        out = paddle.multiply(x, y_t)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("multiply")
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
        out = builder.multiply(x, y, axis=self.case["axis"])

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


class TestElementwiseMulOpBase(TestCaseHelper):
    inputs = [
        {
            "x_shape": [1],
            "y_shape": [1],
            "axis": 0,
        },
        {
            "x_shape": [1024],
            "y_shape": [1024],
            "axis": 0,
        },
        {
            "x_shape": [512, 256],
            "y_shape": [512, 256],
            "axis": 0,
        },
        {
            "x_shape": [128, 64, 32],
            "y_shape": [128, 64, 32],
            "axis": 0,
        },
        {
            "x_shape": [16, 8, 4, 2],
            "y_shape": [16, 8, 4, 2],
            "axis": 0,
        },
        {
            "x_shape": [16, 8, 4, 2, 1],
            "y_shape": [16, 8, 4, 2, 1],
            "axis": 0,
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
        self.class_name = "TestElementwiseMulOpBase"
        self.cls = TestElementwiseMulOp


class TestElementwiseMulOpShapeTest(TestElementwiseMulOpBase):
    def init_attrs(self):
        self.class_name = "TestElementwiseMulOpShapeTest"
        self.cls = TestElementwiseMulOp
        self.inputs = [
            {
                "x_shape": [1],
                "y_shape": [1],
                "axis": 0,
            },
            {
                "x_shape": [1024],
                "y_shape": [1024],
                "axis": -1,
            },
            {
                "x_shape": [2048],
                "y_shape": [2048],
                "axis": 0,
            },
            {
                "x_shape": [512, 256],
                "y_shape": [512, 256],
                "axis": 0,
            },
            {
                "x_shape": [128, 64, 32],
                "y_shape": [128, 64, 32],
                "axis": -1,
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [16, 8, 4, 2],
                "axis": 0,
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "y_shape": [16, 8, 4, 2, 1],
                "axis": -1,
            },
            {
                "x_shape": [1, 1, 1, 1, 1],
                "y_shape": [1, 1, 1, 1, 1],
                "axis": 0,
            },
        ]


class TestElementwiseMulOpDtypeTest(TestElementwiseMulOpBase):
    def init_attrs(self):
        self.class_name = "TestElementwiseMulOpDtypeTest"
        self.cls = TestElementwiseMulOp
        self.dtypes = [
            {
                "x_dtype": "bool",
                "y_dtype": "bool",
            },
            {
                "x_dtype": "int32",
                "y_dtype": "int32",
            },
            {
                "x_dtype": "int64",
                "y_dtype": "int64",
            },
            {
                "x_dtype": "float32",
                "y_dtype": "float32",
            },
            {
                "x_dtype": "float64",
                "y_dtype": "float64",
            },
        ]


class TestElementwiseMulOpPolarityTest(TestElementwiseMulOpBase):
    def init_attrs(self):
        self.class_name = "TestElementwiseMulOpPolarityTest"
        self.cls = TestElementwiseMulOp
        self.attrs = [
            {
                "x_low": -100,
                "x_high": 100,
                "y_low": -100,
                "y_high": 100,
            }
        ]


class TestElementwiseMulOpBroadcast(TestElementwiseMulOpBase):
    def init_attrs(self):
        self.class_name = "TestElementwiseMulOpBroadcast"
        self.cls = TestElementwiseMulOp
        self.inputs = [
            {
                "x_shape": [1],
                "y_shape": [1],
                "axis": 0,
            },
            {
                "x_shape": [1024],
                "y_shape": [1],
                "axis": -1,
            },
            {
                "x_shape": [512, 256],
                "y_shape": [1, 1],
                "axis": 0,
            },
            {
                "x_shape": [128, 64, 32],
                "y_shape": [1, 1, 1],
                "axis": -1,
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [1, 1, 1, 1],
                "axis": 0,
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "y_shape": [1, 1, 1, 1, 1],
                "axis": -1,
            },
        ]


if __name__ == "__main__":
    TestElementwiseMulOpShapeTest().run()
    TestElementwiseMulOpDtypeTest().run()
    TestElementwiseMulOpPolarityTest().run()
    TestElementwiseMulOpBroadcast().run()
