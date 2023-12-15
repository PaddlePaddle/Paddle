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
class TestCholeskyOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if "batch_dim" in self.case and self.case["batch_dim"] > 0:
            x = []
            for _ in range(self.case["batch_dim"]):
                matrix = self.random(
                    self.case["shape"], self.case["dtype"], -1.0, 1.0
                )
                matrix_t = np.transpose(matrix, [1, 0])
                x.append(np.dot(matrix, matrix_t))
            x = np.stack(x)
        else:
            matrix = self.random(
                self.case["shape"], self.case["dtype"], -1.0, 1.0
            )
            matrix_t = np.transpose(matrix, [1, 0])
            x = np.dot(matrix, matrix_t)
        self.inputs = {"x": x}
        self.upper = self.case["upper"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.linalg.cholesky(x, upper=self.upper)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("cholesky")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.cholesky(x, self.upper)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[]
        )
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestCholeskyOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCholeskyOpShape"
        self.cls = TestCholeskyOp
        self.inputs = [
            {
                "shape": [1, 1],
            },
            {
                "shape": [8, 8],
            },
            {
                "shape": [10, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"upper": False},
        ]


class TestCholeskyOpLargeShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCholeskyOpLargeShape"
        self.cls = TestCholeskyOp
        self.inputs = [
            {
                "shape": [1024, 1024],
            },
            {
                "shape": [2048, 2048],
            },
        ]
        self.dtypes = [
            {"dtype": "float64"},
        ]
        self.attrs = [
            {"upper": False, "batch_dim": 2},
            {"upper": False, "batch_dim": 4},
            {"upper": True, "batch_dim": 8},
        ]


class TestCholeskyOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCholeskyOpDtype"
        self.cls = TestCholeskyOp
        self.inputs = [
            {
                "shape": [1, 1],
            },
            {
                "shape": [8, 8],
            },
            {
                "shape": [10, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = [
            {"upper": False},
        ]


class TestCholeskyOpBatch(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCholeskyOpBatch"
        self.cls = TestCholeskyOp
        self.inputs = [
            {
                "shape": [1, 1],
            },
            {
                "shape": [8, 8],
            },
            {
                "shape": [10, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"upper": False, "batch_dim": 1},
            {"upper": False, "batch_dim": 4},
            {"upper": False, "batch_dim": 8},
        ]


class TestCholeskyOpAttrs(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCholeskyOpAttrs"
        self.cls = TestCholeskyOp
        self.inputs = [
            {
                "shape": [1, 1],
            },
            {
                "shape": [8, 8],
            },
            {
                "shape": [10, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = [
            {
                "upper": True,
            },
        ]


if __name__ == "__main__":
    TestCholeskyOpShape().run()
    TestCholeskyOpLargeShape().run()
    TestCholeskyOpDtype().run()
    TestCholeskyOpBatch().run()
    TestCholeskyOpAttrs().run()
