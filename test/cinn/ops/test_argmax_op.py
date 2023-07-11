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
class TestArgMaxOp(OpTest):
    def setUp(self):
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            self.case["shape"], self.case["dtype"], low=0, high=10
        )
        self.axis = self.case["axis"]
        self.keepdim = self.case["keepdim"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.argmax(x, self.axis, self.keepdim)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argmax")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x"
        )
        out = builder.argmax(x, self.axis, self.keepdim)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.x_np], [out]
        )
        self.cinn_outputs = np.array(forward_res).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgMaxOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgMaxOpShapeTest"
        self.cls = TestArgMaxOp
        self.inputs = [
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [1200],
            },
            {
                "shape": [64, 16],
            },
            {
                "shape": [4, 32, 8],
            },
            {
                "shape": [16, 8, 4, 2],
            },
            {
                "shape": [2, 8, 4, 2, 5],
            },
            {
                "shape": [4, 8, 1, 2, 16],
            },
            {
                "shape": [1],
            },
            {
                "shape": [1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1, 1],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axis": 0, "keepdim": False}]


class TestArgMaxOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgMaxOpDtypeTest"
        self.cls = TestArgMaxOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float16",
            },
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
            {
                "dtype": "uint8",
            },
            {
                "dtype": "int16",
            },
            {
                "dtype": "int32",
            },
            {
                "dtype": "int64",
            },
        ]
        self.attrs = [{"axis": 0, "keepdim": False}]


class TestArgMaxOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgMaxOpAxisTest"
        self.cls = TestArgMaxOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "keepdim": False},
            {"axis": 1, "keepdim": False},
            {"axis": 2, "keepdim": False},
            {"axis": 3, "keepdim": False},
        ]


class TestArgMaxOpKeepdimTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgMaxOpKeepdimTest"
        self.cls = TestArgMaxOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "keepdim": True},
            {"axis": 1, "keepdim": True},
            {"axis": 2, "keepdim": True},
            {"axis": 3, "keepdim": True},
        ]


if __name__ == "__main__":
    TestArgMaxOpShapeTest().run()
    TestArgMaxOpDtypeTest().run()
    TestArgMaxOpAxisTest().run()
    TestArgMaxOpKeepdimTest().run()
