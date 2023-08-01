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
from op_test_helper import TestCaseHelper, run_test

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random(self.case["shape1"], self.case["dtype"]),
            "input2": self.random(self.case["shape2"], self.case["dtype"]),
        }
        self.left_side = self.case["left_side"]
        self.upper = self.case["upper"]
        self.transpose_a = self.case["transpose_a"]
        self.unit_diagonal = self.case["unit_diagonal"]

    def build_paddle_program(self, target):
        def transpose_last_two_dims(x):
            shape = x.shape
            last_dim_idx = len(shape) - 1
            second_last_dim_idx = len(shape) - 2
            perm = list(range(len(shape)))
            perm[last_dim_idx], perm[second_last_dim_idx] = (
                perm[second_last_dim_idx],
                perm[last_dim_idx],
            )
            x_transposed = paddle.transpose(x, perm=perm)
            return x_transposed

        input1 = paddle.to_tensor(self.inputs["input1"], stop_gradient=True)
        input2 = paddle.to_tensor(self.inputs["input2"], stop_gradient=True)
        if self.left_side:
            out = paddle.linalg.triangular_solve(
                input1, input2, self.upper, self.transpose_a, self.unit_diagonal
            )
            self.paddle_outputs = [out]
        else:
            input1 = transpose_last_two_dims(input1)
            input2 = transpose_last_two_dims(input2)
            out = paddle.linalg.triangular_solve(
                input1,
                input2,
                not self.upper,
                self.transpose_a,
                self.unit_diagonal,
            )
            out = transpose_last_two_dims(out)
            self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("triangular_solve")
        input1 = builder.create_input(
            self.nptype2cinntype(self.inputs["input1"].dtype),
            self.inputs["input1"].shape,
            "input1",
        )
        input2 = builder.create_input(
            self.nptype2cinntype(self.inputs["input2"].dtype),
            self.inputs["input2"].shape,
            "input2",
        )
        out = builder.triangular_solve(
            input1,
            input2,
            self.left_side,
            self.upper,
            self.transpose_a,
            self.unit_diagonal,
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [input1, input2],
            [self.inputs["input1"], self.inputs["input2"]],
            [out],
            passes=[],
        )
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTriangularSolveOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTriangularSolveOpShapeTest"
        self.cls = TestTriangularSolveOp
        self.inputs = [
            {
                "shape1": [1, 3, 3],
                "shape2": [1, 3, 1],
            },
            {
                "shape1": [1, 1, 1],
                "shape2": [1, 1, 1],
            },
            {
                "shape1": [2, 3, 3],
                "shape2": [2, 3, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": False,
            },
        ]


class TestTriangularSolveOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTriangularSolveOpDtypeTest"
        self.cls = TestTriangularSolveOp
        self.inputs = [
            {
                "shape1": [2, 8, 8],
                "shape2": [2, 8, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = [
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": False,
            },
        ]


class TestTriangularSolveOpBatchDimTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTriangularSolveOpBatchDimTest"
        self.cls = TestTriangularSolveOp
        self.inputs = [
            {
                "shape1": [8, 8],
                "shape2": [8, 4],
            },
            {
                "shape1": [3, 16, 16],
                "shape2": [16, 4],
            },
            {
                "shape1": [16, 16],
                "shape2": [5, 16, 4],
            },
            {
                "shape1": [5, 16, 16],
                "shape2": [5, 16, 4],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": False,
            },
        ]


class TestTriangularSolveOpBroadcastTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTriangularSolveOpBroadcastTest"
        self.cls = TestTriangularSolveOp
        self.inputs = [
            {
                "shape1": [2, 2, 3, 3, 3],
                "shape2": [1, 3, 4],
            },
            {
                "shape1": [3, 3, 3],
                "shape2": [2, 2, 3, 3, 4],
            },
            {
                "shape1": [2, 1, 3, 3, 3],
                "shape2": [2, 2, 3, 3, 4],
            },
            {
                "shape1": [5, 1, 3, 3, 3],
                "shape2": [1, 2, 1, 3, 4],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": False,
            },
        ]


class TestTriangularSolveOpAttributeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTriangularSolveOpAttributeTest"
        self.cls = TestTriangularSolveOp
        self.inputs = [
            {
                "shape1": [1, 3, 3],
                "shape2": [1, 3, 1],
            },
            {
                "shape1": [2, 3, 3],
                "shape2": [2, 3, 10],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": False,
            },
            {
                "left_side": True,
                "upper": True,
                "transpose_a": False,
                "unit_diagonal": True,
            },
            {
                "left_side": True,
                "upper": True,
                "transpose_a": True,
                "unit_diagonal": False,
            },
        ]


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpRightSide(TestTriangularSolveOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([2, 3, 3], "float32"),
            "input2": self.random([2, 1, 3], "float32"),
        }
        self.left_side = False
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


class TestTriangularSolveOpRightSide1(TestTriangularSolveOpRightSide):
    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([1, 3, 2, 3, 3], "float32"),
            "input2": self.random([2, 1, 2, 1, 3], "float32"),
        }
        self.left_side = False
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpSingular(TestTriangularSolveOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([1, 3, 3], "float32"),
            "input2": self.random([1, 3, 1], "float32"),
        }
        # set one dim to zeros to make a singular matrix
        self.inputs["input1"][0][0] = 0
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


class TestTriangularSolveOpSingular1(TestTriangularSolveOpSingular):
    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([1, 3, 3], "float32"),
            "input2": self.random([1, 3, 1], "float32"),
        }
        # set one dim to zeros to make a singular matrix
        self.inputs["input1"][0][2] = 0
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpLarge(TestTriangularSolveOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([1, 1024, 1024], "float64", -0.01, 0.01),
            "input2": self.random([1, 1024, 512], "float64", -0.01, 0.01),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


class TestTriangularSolveOpLarge1(TestTriangularSolveOpLarge):
    def prepare_inputs(self):
        self.inputs = {
            "input1": self.random([1, 2048, 2048], "float64", -0.01, 0.01),
            "input2": self.random([1, 2048, 512], "float64", -0.01, 0.01),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


if __name__ == "__main__":
    TestTriangularSolveOpShapeTest().run()
    TestTriangularSolveOpDtypeTest().run()
    TestTriangularSolveOpBatchDimTest().run()
    TestTriangularSolveOpBroadcastTest().run()
    TestTriangularSolveOpAttributeTest().run()

    run_test(TestTriangularSolveOpRightSide)
    run_test(TestTriangularSolveOpRightSide1)
    run_test(TestTriangularSolveOpSingular)
    run_test(TestTriangularSolveOpSingular1)
    run_test(TestTriangularSolveOpLarge)
    run_test(TestTriangularSolveOpLarge1)
