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
class TestSelectOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "Condition": self.random(self.case["shape"], "bool"),
            "X": self.random(self.case["shape"], self.case["dtype"]),
            "Y": self.random(self.case["shape"], self.case["dtype"]),
        }

    def build_paddle_program(self, target):
        c = paddle.to_tensor(self.inputs["Condition"], stop_gradient=True)
        x = paddle.to_tensor(self.inputs["X"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["Y"], stop_gradient=True)

        out = paddle.where(c, x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("select")
        c = builder.create_input(
            self.nptype2cinntype(self.inputs["Condition"].dtype),
            self.inputs["Condition"].shape,
            "Condition",
        )
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["X"].dtype),
            self.inputs["X"].shape,
            "X",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["Y"].dtype),
            self.inputs["Y"].shape,
            "Y",
        )

        out = builder.select(c, x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [c, x, y],
            [self.inputs["Condition"], self.inputs["X"], self.inputs["Y"]],
            [out],
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSelectOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSelectOpShape"
        self.cls = TestSelectOp
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


class TestSelectOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSelectOpDtype"
        self.cls = TestSelectOp
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
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestSelectOpShape().run()
    TestSelectOpDtype().run()
