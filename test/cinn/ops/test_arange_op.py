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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestArangeOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "start": self.case["start"],
            "end": self.case["end"],
            "step": self.case["step"],
            "dtype": self.case["dtype"],
        }

    def build_paddle_program(self, target):
        out = paddle.arange(
            self.inputs["start"],
            self.inputs["end"],
            self.inputs["step"],
            self.inputs["dtype"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("arange")
        out = builder.arange(
            self.inputs["start"],
            self.inputs["end"],
            self.inputs["step"],
            self.inputs["dtype"],
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestArangeOpShapeAndAttr(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestArangeOpShapeAndAttr"
        self.cls = TestArangeOp
        self.inputs = [
            # basic shape test
            {
                "start": 0,
                "end": 10,
                "step": 1,
            },
            {
                "start": 0,
                "end": 1024,
                "step": 16,
            },
            {
                "start": 512,
                "end": 2600,
                "step": 512,
            },
            {
                "start": 0,
                "end": 65536,
                "step": 1024,
            },
            {
                "start": 0,
                "end": 131072,
                "step": 2048,
            },
            {
                "start": 0,
                "end": 1,
                "step": 2,
            },
            {
                "start": 0,
                "end": 1,
                "step": 2,
            },
            # step test
            {
                "start": 1024,
                "end": 512,
                "step": -2,
            },
            {
                "start": 2048,
                "end": 0,
                "step": -64,
            },
            # range test
            {
                "start": -2048,
                "end": 2048,
                "step": 32,
            },
            {
                "start": -2048,
                "end": -512,
                "step": 64,
            },
            {
                "start": 1024,
                "end": 4096,
                "step": 512,
            },
            {
                "start": 1024,
                "end": -1024,
                "step": -128,
            },
            {
                "start": -1024,
                "end": -2048,
                "step": -64,
            },
            {
                "start": 2048,
                "end": 512,
                "step": -32,
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestArangeOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestArangeOpDtype"
        self.cls = TestArangeOp
        self.inputs = [
            {
                "start": 5,
                "end": 10,
                "step": 1,
            },
            {
                "start": -10,
                "end": -100,
                "step": -10,
            },
            {
                "start": -10,
                "end": 10,
                "step": 1,
            },
        ]
        self.dtypes = [
            {"dtype": "int32"},
            {"dtype": "int64"},
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestArangeOpShapeAndAttr().run()
    TestArangeOpDtype().run()
