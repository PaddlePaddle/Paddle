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
class TestFillConstantOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.shape = self.case["shape"]
        self.value = self.case["value"]
        self.dtype = self.case["dtype"]
        if isinstance(self.value, str):
            dtypes = ["bool", "int", "float"]
            for dtype in dtypes:
                if dtype in self.dtype:
                    try:
                        self.value = eval(f"{dtype}(self.value)")
                    except:
                        self.value = eval(f"{dtype}(0)")

    def build_paddle_program(self, target):
        if self.dtype is None:
            x = np.full(self.shape, self.value)
            x = paddle.to_tensor(x)
        else:
            x = paddle.full(self.shape, self.value, dtype=self.dtype)

        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("fill_constant")
        if self.dtype is None:
            x = builder.fill_constant(self.shape, self.value, "out")
        else:
            x = builder.fill_constant(self.shape, self.value, "out", self.dtype)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestFillConstantOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestFillConstantOpShape"
        self.cls = TestFillConstantOp
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
                "shape": [1, 2, 4, 8],
            },
            {
                "shape": [16, 4, 8, 32],
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
        self.attrs = [
            {"value": 123.456},
        ]


class TestFillConstantOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestFillConstantOpDtype"
        self.cls = TestFillConstantOp
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
                "shape": [1, 2, 4, 8],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "bool"},
            {"dtype": "uint8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [
            {"value": 123.456},
        ]


class TestFillConstantOpValue(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestFillConstantOpValue"
        self.cls = TestFillConstantOp
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
                "shape": [1, 2, 4, 8],
            },
        ]
        self.dtypes = [
            {"dtype": None},
        ]
        self.attrs = [
            {"value": True},
            {"value": 123},
            {"value": 123.456},
        ]


class TestFillConstantOpStrValue(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestFillConstantOpStrValue"
        self.cls = TestFillConstantOp
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
                "shape": [1, 2, 4, 8],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "bool"},
            {"dtype": "uint8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [
            {"value": "1024"},
            {"value": "0.12345678987654321"},
        ]


if __name__ == "__main__":
    TestFillConstantOpShape().run()
    TestFillConstantOpDtype().run()
    TestFillConstantOpValue().run()
    TestFillConstantOpStrValue().run()
