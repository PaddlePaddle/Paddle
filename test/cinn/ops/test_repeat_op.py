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
class TestRepeatOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        shape = self.case["shape"]
        dtype = self.case["dtype"]
        repeats = self.case["repeats"]
        axis = self.case["axis"]
        dims = len(shape)
        axis = min(axis, dims - 1)
        axis = max(axis, -dims)
        self.inputs = {
            "x": self.random(shape, dtype, -1.0, 1.0),
            "repeats": repeats,
            "axis": axis,
        }

    def build_paddle_program(self, target):
        x = np.repeat(
            self.inputs["x"], self.inputs["repeats"], self.inputs["axis"]
        )
        out = paddle.to_tensor(x, stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("repeat")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.repeat(x, self.inputs["repeats"], self.inputs["axis"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestRepeatOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRepeatOpShape"
        self.cls = TestRepeatOp
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
        self.attrs = [
            {"repeats": 2, "axis": 0},
        ]


class TestRepeatOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRepeatOpDtype"
        self.cls = TestRepeatOp
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
            {"dtype": "bool"},
            {"dtype": "int8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = [
            {"repeats": 4, "axis": 0},
        ]


class TestRepeatOpAttributeRepeats(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRepeatOpAttributeRepeats"
        self.cls = TestRepeatOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"repeats": 256, "axis": 0},
            {"repeats": 1024, "axis": 0},
            {"repeats": 2048, "axis": 0},
        ]


class TestRepeatOpAttributeAxis(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRepeatOpAttributeAxis"
        self.cls = TestRepeatOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"repeats": 128, "axis": 0},
            {"repeats": 128, "axis": 1},
            {"repeats": 128, "axis": 2},
            {"repeats": 128, "axis": 3},
            {"repeats": 128, "axis": -1},
            {"repeats": 128, "axis": -2},
            {"repeats": 128, "axis": -3},
            {"repeats": 128, "axis": -4},
        ]


if __name__ == "__main__":
    TestRepeatOpShape().run()
    TestRepeatOpDtype().run()
    TestRepeatOpAttributeRepeats().run()
    TestRepeatOpAttributeAxis().run()
