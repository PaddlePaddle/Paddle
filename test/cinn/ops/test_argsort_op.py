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
class TestArgSortOp(OpTest):
    def setUp(self):
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.case["shape"], self.case["dtype"])
        self.axis = self.case["axis"]
        self.descending = self.case["descending"]

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.argsort(x1, self.axis, self.descending)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argsort")
        x1 = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x1"
        )
        out = builder.argsort(x1, self.axis, not self.descending)
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x1], [self.x_np], out)
        self.cinn_outputs = np.array([forward_res[0]]).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgSortOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpShapeTest"
        self.cls = TestArgSortOp
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
        self.attrs = [{"axis": 0, "descending": False}]


class TestArgSortOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpDtypeTest"
        self.cls = TestArgSortOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
            # Throw dtype not support error in paddle
            # {
            #     "dtype": "uint8",
            # },
            {
                "dtype": "int32",
            },
            {
                "dtype": "int64",
            },
        ]
        self.attrs = [{"axis": 0, "descending": False}]


class TestArgSortOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpAxisTest"
        self.cls = TestArgSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "descending": False},
            {"axis": 1, "descending": False},
            {"axis": 2, "descending": False},
            {"axis": 3, "descending": False},
        ]


class TestArgSortOpDescedingTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpDescedingTest"
        self.cls = TestArgSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "descending": True},
            {"axis": 1, "descending": True},
            {"axis": 2, "descending": True},
            {"axis": 3, "descending": True},
        ]


if __name__ == "__main__":
    TestArgSortOpShapeTest().run()
    TestArgSortOpDtypeTest().run()
    TestArgSortOpAxisTest().run()
    TestArgSortOpDescedingTest().run()
