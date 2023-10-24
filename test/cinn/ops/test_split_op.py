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
class TestSplitOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.case["shape"], self.case["dtype"], -1.0, 1.0)
        }
        self.num_or_sections = self.case["num_or_sections"]
        self.axis = self.case["axis"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        if len(self.num_or_sections) == 1:
            num = self.num_or_sections[0]
        else:
            num = self.num_or_sections
        out = paddle.split(x, num_or_sections=num, axis=self.axis)
        self.paddle_outputs = out

    def build_cinn_program(self, target):
        builder = NetBuilder("split")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.split(
            x, num_or_sections=self.num_or_sections, axis=self.axis
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], out)
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSplitOpLegacy(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpLegacy"
        self.cls = TestSplitOp
        self.inputs = [
            {"shape": [9, 9, 5], "num_or_sections": [2, 3, 4], "axis": 0},
            {"shape": [9, 9, 5], "num_or_sections": [3], "axis": 0},
            {"shape": [9, 9, 5], "num_or_sections": [3], "axis": 1},
            {"shape": [9, 9, 5], "num_or_sections": [2, 3, -1], "axis": 1},
            {"shape": [8, 9, 5], "num_or_sections": [2], "axis": 0},
            {"shape": [8, 9, 5], "num_or_sections": [-1, 2, 2, 2], "axis": 0},
            {"shape": [2048, 9, 6], "num_or_sections": [2], "axis": 2},
            {"shape": [10, 128, 4096], "num_or_sections": [2], "axis": 2},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestSplitOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpShape"
        self.cls = TestSplitOp
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
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            {
                "shape": [2048],
            },
            {
                "shape": [65536],
            },
            {
                "shape": [131072],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [{"num_or_sections": [2], "axis": 0}]


class TestSplitOpOnes(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpOnes"
        self.cls = TestSplitOp
        self.inputs = [
            {"shape": [1], "num_or_sections": [1], "axis": 0},
            {"shape": [1], "num_or_sections": [1], "axis": 0},
            {"shape": [1, 1, 1, 1], "num_or_sections": [1], "axis": 0},
            {"shape": [1, 1, 1, 1], "num_or_sections": [1], "axis": 2},
            {"shape": [1, 1, 1, 1, 1], "num_or_sections": [1], "axis": 4},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestSplitOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpDtype"
        self.cls = TestSplitOp
        self.inputs = [
            {
                "shape": [8],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [{"num_or_sections": [2], "axis": 0}]


class TestSplitOpAttributeNum(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpAttributeNum"
        self.cls = TestSplitOp
        self.inputs = [
            {"shape": [1024], "num_or_sections": [16], "axis": 0},
            {"shape": [1024], "num_or_sections": [-1, 256, 256], "axis": 0},
            {"shape": [256, 32], "num_or_sections": [-1, 16], "axis": 1},
            {"shape": [16, 8, 32, 64], "num_or_sections": [2, 3, 3], "axis": 1},
            {
                "shape": [1, 1, 1, 16, 1],
                "num_or_sections": [4, 4, 4, 4],
                "axis": 3,
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestSplitOpAttributeAxis(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpAttributeAxis"
        self.cls = TestSplitOp
        self.inputs = [
            {
                "shape": [16, 8, 32, 64],
                "num_or_sections": [3, -1, 3],
                "axis": 0,
            },
            {
                "shape": [16, 8, 32, 64],
                "num_or_sections": [3, -1, 3],
                "axis": 1,
            },
            {
                "shape": [16, 8, 32, 64],
                "num_or_sections": [3, -1, 3],
                "axis": 2,
            },
            {
                "shape": [16, 8, 32, 64],
                "num_or_sections": [3, -1, 3],
                "axis": 3,
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestSplitOpAttributeLargeNum(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSplitOpAttributeLargeNum"
        self.cls = TestSplitOp
        self.inputs = [
            {"shape": [1024], "num_or_sections": [16], "axis": 0},
            {"shape": [1024], "num_or_sections": [256], "axis": 0},
            {"shape": [1024], "num_or_sections": [1024], "axis": 0},
            {"shape": [1024], "num_or_sections": [512], "axis": 0},
            {"shape": [131072], "num_or_sections": [131072], "axis": 0},
            {"shape": [131072], "num_or_sections": [65536], "axis": 0},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestSplitOpLegacy().run()
    TestSplitOpShape().run()
    TestSplitOpOnes().run()
    TestSplitOpDtype().run()
    TestSplitOpAttributeNum().run()
    TestSplitOpAttributeAxis().run()
    # TestSplitOpAttributeLargeNum().run()
