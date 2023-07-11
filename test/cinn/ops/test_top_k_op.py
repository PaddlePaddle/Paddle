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
from op_test_helper import TestCaseHelper, run_test

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestTopKOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.case["shape"], self.case["dtype"])}
        self.k = self.case["k"]
        self.axis = self.case["axis"]
        self.largest = self.case["largest"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        if x.shape[self.axis] < self.k:
            self.k = x.shape[self.axis]
        out = paddle.topk(x, self.k, self.axis)
        self.paddle_outputs = [out[0], out[1]]

    def build_cinn_program(self, target):
        builder = NetBuilder("topk")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.top_k(x, self.k, self.axis, self.largest)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out[0], out[1]]
        )
        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTopKOpDumpicateElement(TestTopKOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random([128], "int64", -10, 10)}
        self.axis = 0
        self.largest = False
        self.k = 5


# known issue: same as sort op
# This test case will cause CINN to allocate a large amount of GPU memory, nearly 10 GB.
class TestTopKOpLargeCudaMemoryOccupation(TestTopKOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random([8192], "float64")}
        self.axis = 0
        self.largest = False
        self.k = 5


class TestTopKOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTopKOpShapeTest"
        self.cls = TestTopKOp
        self.inputs = [
            {"shape": [512], "k": 3},
            {"shape": [1024], "k": 10},
            {"shape": [1200], "k": 1024},
            {"shape": [64, 16], "k": 3},
            {"shape": [4, 32, 8], "k": 4},
            {"shape": [16, 8, 4, 2], "k": 5},
            {"shape": [2, 8, 4, 2, 5], "k": 1},
            {"shape": [4, 8, 1, 2, 16], "k": 3},
            {"shape": [1], "k": 1},
            {"shape": [1, 1, 1, 1], "k": 1},
            {"shape": [1, 1, 1, 1, 1], "k": 1},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axis": 0, "largest": True}]


class TestTopKOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTopKOpDtypeTest"
        self.cls = TestTopKOp
        self.inputs = [
            {
                "shape": [1024],
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
        ]
        self.dtypes = [
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [{"axis": 0, "largest": True, "k": 3}]


class TestTopKOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTopKOpAxisTest"
        self.cls = TestTopKOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 8],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "largest": True, "k": 3},
            {"axis": 1, "largest": True, "k": 3},
            {"axis": 2, "largest": True, "k": 3},
            {"axis": 3, "largest": True, "k": 3},
        ]


class TestTopKOpKTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTopKOpKTest"
        self.cls = TestTopKOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "largest": True, "k": 8},
            {"axis": 1, "largest": True, "k": 4},
            {"axis": 2, "largest": True, "k": 2},
            {"axis": 3, "largest": True, "k": 1},
            {"axis": 0, "largest": True, "k": 20},
            {"axis": 1, "largest": True, "k": 10},
            {"axis": 2, "largest": True, "k": 10},
            {"axis": 3, "largest": True, "k": 5},
        ]


class TestTopKOpAscendingTest(TestTopKOpShapeTest):
    def init_attrs(self):
        self.class_name = "TestTopKOpAscendingTest"
        self.cls = TestTopKOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 8],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "largest": False, "k": 3},
            {"axis": 1, "largest": False, "k": 3},
            {"axis": 2, "largest": False, "k": 3},
            {"axis": 3, "largest": False, "k": 3},
        ]


if __name__ == "__main__":
    run_test(TestTopKOpDumpicateElement)
    # run_test(TestTopKOpLargeCudaMemoryOccupation)

    TestTopKOpShapeTest().run()
    TestTopKOpDtypeTest().run()
    TestTopKOpAxisTest().run()
    TestTopKOpKTest().run()
    TestTopKOpAscendingTest().run()
