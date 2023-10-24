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
class TestScatterAssignOpBase(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs["x"] = self.random(self.case["x_shape"]).astype(
            self.case["x_dtype"]
        )
        self.inputs["y"] = self.random(self.case["y_shape"]).astype(
            self.case["y_dtype"]
        )
        self.inputs["index"] = np.random.randint(
            0, self.case["index_upper"], size=self.case["index_size"]
        ).astype("int32")
        self.axis = self.case["axis"]

    def build_paddle_program(self, target):
        x = self.inputs["x"].copy()
        y = self.inputs["y"].copy()

        out = x
        axis = self.axis
        while axis < 0:
            axis += len(self.inputs["x"].shape)

        if axis == 0:
            for i in range(self.inputs["index"].shape[0]):
                out[self.inputs["index"][i]] = y[i]
        elif axis == 1:
            for i in range(self.inputs["x"].shape[0]):
                for j in range(self.inputs["index"].shape[0]):
                    out[i][self.inputs["index"][j]] = y[i][j]
        elif axis == 2:
            for i in range(self.inputs["x"].shape[0]):
                for j in range(self.inputs["x"].shape[1]):
                    for k in range(self.inputs["index"].shape[0]):
                        out[i][j][self.inputs["index"][k]] = y[i][j][k]
        elif axis == 3:
            for i in range(self.inputs["x"].shape[0]):
                for j in range(self.inputs["x"].shape[1]):
                    for k in range(self.inputs["x"].shape[2]):
                        for l in range(self.inputs["index"].shape[0]):
                            out[i][j][k][self.inputs["index"][l]] = y[i][j][k][
                                l
                            ]
        else:
            self.assertTrue(False, f"Axis {self.axis} No Implement")

        pd_out = paddle.to_tensor(out, stop_gradient=True)
        self.paddle_outputs = [pd_out]

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_assign")
        x = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        index = builder.create_input(
            OpTest.nptype2cinntype(self.inputs["index"].dtype),
            self.inputs["index"].shape,
            "index",
        )
        out = builder.scatter_assign(x, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [self.inputs["x"], self.inputs["y"], self.inputs["index"]],
            [out],
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestScatterAssignOp(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAssignOp"
        self.cls = TestScatterAssignOpBase
        self.inputs = [
            {
                "x_shape": [10],
                "y_shape": [1],
                "index_upper": 10,
                "index_size": 1,
                "axis": -1,
            },
            {
                "x_shape": [10, 5],
                "y_shape": [3, 5],
                "index_upper": 10,
                "index_size": 3,
                "axis": 0,
            },
            {
                "x_shape": [10, 5, 5],
                "y_shape": [10, 5, 4],
                "index_upper": 5,
                "index_size": 4,
                "axis": -1,
            },
            {
                "x_shape": [10, 5, 5, 7],
                "y_shape": [10, 5, 2, 7],
                "index_upper": 5,
                "index_size": 2,
                "axis": -2,
            },
            {
                "x_shape": [10, 5, 1024, 2048],
                "y_shape": [10, 5, 2, 2048],
                "index_upper": 5,
                "index_size": 2,
                "axis": -2,
            },
        ]
        self.dtypes = [
            {"x_dtype": "float32", "y_dtype": "float32"},
        ]
        self.attrs = []


class TestScatterAssignOpAttribute(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAssignOpAttribute"
        self.cls = TestScatterAssignOpBase
        self.inputs = [
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [1, 1, 1, 1],
                "index_upper": 1,
                "index_size": 1,
                "axis": 0,
            },
            {
                "x_shape": [1, 10, 10, 3],
                "y_shape": [1, 4, 10, 3],
                "index_upper": 10,
                "index_size": 4,
                "axis": 1,
            },
            {
                "x_shape": [10, 4, 8, 3],
                "y_shape": [10, 4, 5, 3],
                "index_upper": 8,
                "index_size": 5,
                "axis": 2,
            },
            {
                "x_shape": [10, 4, 5, 6],
                "y_shape": [10, 4, 5, 3],
                "index_upper": 6,
                "index_size": 3,
                "axis": 3,
            },
            {
                "x_shape": [10, 4, 5, 1024],
                "y_shape": [10, 4, 5, 768],
                "index_upper": 1024,
                "index_size": 768,
                "axis": -1,
            },
            {
                "x_shape": [1024, 4, 12, 10],
                "y_shape": [1024, 4, 5, 10],
                "index_upper": 12,
                "index_size": 5,
                "axis": -2,
            },
            {
                "x_shape": [10, 8192, 12, 10],
                "y_shape": [10, 4096, 12, 10],
                "index_upper": 8192,
                "index_size": 4096,
                "axis": -3,
            },
            {
                "x_shape": [2048, 10, 12, 10],
                "y_shape": [1024, 10, 12, 10],
                "index_upper": 2048,
                "index_size": 1024,
                "axis": -4,
            },
        ]
        self.dtypes = [
            {"x_dtype": "float32", "y_dtype": "float32"},
        ]
        self.attrs = []


class TestScatterAssignOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAssignOpDtype"
        self.cls = TestScatterAssignOpBase
        self.inputs = [
            {
                "x_shape": [10, 5, 20, 7],
                "y_shape": [10, 5, 15, 7],
                "index_upper": 20,
                "index_size": 15,
                "axis": -2,
            },
        ]
        self.dtypes = [
            {"x_dtype": "float16", "y_dtype": "float16"},
            {"x_dtype": "float32", "y_dtype": "float32"},
            {"x_dtype": "float64", "y_dtype": "float64"},
            {"x_dtype": "int32", "y_dtype": "int32"},
            {"x_dtype": "int64", "y_dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestScatterAssignOp().run()
    TestScatterAssignOpAttribute().run()
    TestScatterAssignOpDtype().run()
