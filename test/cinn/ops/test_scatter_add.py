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

from cinn.common import Float, Int, is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper, run_test

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestScatterAddOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        x_shape = self.case["x_shape"]
        y_shape = self.case["y_shape"]
        dtype = self.case["dtype"]
        axis = self.case["axis"]
        self.inputs = {
            "x": self.random(x_shape, dtype),
            "y": self.random(y_shape, dtype),
            "index": self.random([y_shape[axis]], "int32", 0, x_shape[axis]),
        }
        self.axis = axis

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        pos_axis = self.axis
        if pos_axis < 0:
            pos_axis += len(x.shape)

        index_nd = []
        if pos_axis == 0:
            for i in range(len(self.inputs["index"])):
                index_nd.append([self.inputs["index"][i]])
        elif pos_axis == 1:
            for i in range(self.inputs['x'].shape[0]):
                index_nd.append([])
                for j in range(len(self.inputs["index"])):
                    index_nd[i].append([i, self.inputs["index"][j]])
        elif pos_axis == 2:
            for i in range(self.inputs['x'].shape[0]):
                index_nd.append([])
                for j in range(self.inputs['x'].shape[1]):
                    index_nd[i].append([])
                    for k in range(len(self.inputs["index"])):
                        index_nd[i][j].append([i, j, self.inputs["index"][k]])
        elif pos_axis == 3:
            for i in range(self.inputs['x'].shape[0]):
                index_nd.append([])
                for j in range(self.inputs['x'].shape[1]):
                    index_nd[i].append([])
                    for k in range(self.inputs['x'].shape[2]):
                        index_nd[i][j].append([])
                        for l in range(len(self.inputs["index"])):
                            index_nd[i][j][k].append(
                                [i, j, k, self.inputs["index"][l]]
                            )
        else:
            self.assertTrue(False, f"Axis {pos_axis} No Implement")

        index = paddle.to_tensor(index_nd, stop_gradient=True)
        res = paddle.scatter_nd_add(x, index, y)
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        index = builder.create_input(
            self.nptype2cinntype(self.inputs["index"].dtype),
            self.inputs["index"].shape,
            "index",
        )
        out = builder.scatter_add(x, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [self.inputs["x"], self.inputs["y"], self.inputs["index"]],
            [out],
        )

        self.cinn_outputs = res

    def test_check_results(self):
        if self.case["dtype"] == "float16":
            self.check_outputs_and_grads(
                max_relative_error=0.01, max_absolute_error=0.01
            )
        else:
            self.check_outputs_and_grads()


class TestScatterAddOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAddOpShapeTest"
        self.cls = TestScatterAddOp
        self.inputs = [
            {"x_shape": [10], "y_shape": [5], "axis": 0},
            {"x_shape": [10, 8], "y_shape": [8, 8], "axis": 0},
            {"x_shape": [10, 8, 16], "y_shape": [10, 4, 16], "axis": 1},
            {
                "x_shape": [10, 8, 16, 32],
                "y_shape": [10, 8, 20, 32],
                "axis": -2,
            },
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 8, 1, 32], "axis": -2},
            {"x_shape": [10, 1, 16, 32], "y_shape": [10, 1, 8, 32], "axis": -2},
            {
                "x_shape": [1024, 8, 16, 4],
                "y_shape": [512, 8, 16, 4],
                "axis": 0,
            },
            {
                "x_shape": [2048, 8, 16, 4],
                "y_shape": [1024, 8, 16, 4],
                "axis": 0,
            },
            {
                "x_shape": [1024, 8, 16, 4],
                "y_shape": [2048, 8, 16, 4],
                "axis": 0,
            },
            {"x_shape": [1, 1, 1, 1], "y_shape": [1, 1, 1, 1], "axis": 0},
            {"x_shape": [1], "y_shape": [8], "axis": 0},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestScatterAddOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAddOpDtypeTest"
        self.cls = TestScatterAddOp
        self.inputs = [
            {"x_shape": [10], "y_shape": [5], "axis": 0},
            {"x_shape": [10, 8], "y_shape": [8, 8], "axis": 0},
            {
                "x_shape": [1024, 8, 16, 4],
                "y_shape": [512, 8, 16, 4],
                "axis": 0,
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = []


class TestScatterAddOpAttributeAxis(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestScatterAddOpAttributeAxis"
        self.cls = TestScatterAddOp
        self.inputs = [
            {"x_shape": [10], "y_shape": [5], "axis": 0},
            {"x_shape": [10, 8], "y_shape": [8, 8], "axis": -2},
            {"x_shape": [10, 8, 16], "y_shape": [5, 8, 16], "axis": 0},
            {"x_shape": [10, 8, 16], "y_shape": [10, 4, 16], "axis": 1},
            {"x_shape": [10, 8, 16], "y_shape": [10, 8, 8], "axis": 2},
            {"x_shape": [10, 8, 16, 32], "y_shape": [2, 8, 16, 32], "axis": 0},
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 8, 8, 32], "axis": 2},
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 8, 16, 16], "axis": 3},
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 8, 16, 8], "axis": -1},
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 8, 4, 32], "axis": -2},
            {"x_shape": [10, 8, 16, 32], "y_shape": [1, 8, 16, 32], "axis": -4},
            {"x_shape": [10, 8, 16, 32], "y_shape": [10, 4, 16, 32], "axis": 1},
            {
                "x_shape": [10, 8, 16, 32],
                "y_shape": [10, 2, 16, 32],
                "axis": -3,
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


# test inline compute: https://github.com/PaddlePaddle/CINN/pull/1329
class TestScatterAddCaseInline1(TestScatterAddOp):
    def setUp(self):
        self.case = {
            "x_shape": [10, 5],
            "y_shape": [5, 5],
            "index_shape": [5],
            "dtype": "float32",
            "index_dtype": "int32",
            "axis": 0,
        }
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(64), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        x1 = builder.cast(x, dtype="float32")  # newly added
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x1, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [
                self.inputs["x"].astype("float64"),
                self.inputs["y"],
                self.inputs["index"],
            ],
            [out],
        )

        self.cinn_outputs = [res[0]]


class TestScatterAddCaseInline2(TestScatterAddCaseInline1):
    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(64), self.inputs["y"].shape, "y")
        y1 = builder.cast(y, dtype="float32")  # newly added
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x, y1, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [
                self.inputs["x"],
                self.inputs["y"].astype("float64"),
                self.inputs["index"],
            ],
            [out],
        )

        self.cinn_outputs = res


if __name__ == "__main__":
    TestScatterAddOpShapeTest().run()
    TestScatterAddOpDtypeTest().run()
    TestScatterAddOpAttributeAxis().run()
    run_test(TestScatterAddCaseInline1)
    run_test(TestScatterAddCaseInline2)
