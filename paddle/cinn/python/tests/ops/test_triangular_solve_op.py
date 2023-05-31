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

import unittest

import numpy as np
from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False

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


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpUnitDiagonal(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = True


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpLower(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = False
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim1(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((3, 3)).astype(np.float32),
            "input2": np.random.random((3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim2(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpZeroBatchDim3(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBroadCast(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((2, 2, 3, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 4)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBroadCast1(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((3, 3, 3)).astype(np.float32),
            "input2": np.random.random((2, 2, 3, 3, 4)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBroadCast2(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((2, 1, 3, 3, 3)).astype(np.float32),
            "input2": np.random.random((2, 2, 3, 3, 4)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBroadCast3(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((5, 1, 3, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 2, 1, 3, 4)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpTranspose(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpRightSide(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((2, 3, 3)).astype(np.float32),
            "input2": np.random.random((2, 1, 3)).astype(np.float32),
        }
        self.left_side = False
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpRightSide1(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 2, 3, 3)).astype(np.float32),
            "input2": np.random.random((2, 1, 2, 1, 3)).astype(np.float32),
        }
        self.left_side = False
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpDoubleFloat(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float64),
            "input2": np.random.random((1, 3, 1)).astype(np.float64),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpBatch(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((5, 3, 3)).astype(np.float32),
            "input2": np.random.random((5, 3, 1)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpMultipleRightHandSides(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((2, 3, 3)).astype(np.float32),
            "input2": np.random.random((2, 3, 10)).astype(np.float32),
        }
        self.left_side = True
        self.upper = True
        self.transpose_a = False
        self.unit_diagonal = False


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpSingular(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        # set one dim to zeros to make a singular matrix
        self.inputs["input1"][0][0] = 0
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "triangular solve op support GPU only now."
)
class TestTriangularSolveOpSingular1(TestTriangularSolveOp):
    def init_case(self):
        self.inputs = {
            "input1": np.random.random((1, 3, 3)).astype(np.float32),
            "input2": np.random.random((1, 3, 1)).astype(np.float32),
        }
        # set one dim to zeros to make a singular matrix
        self.inputs["input1"][0][2] = 0
        self.left_side = True
        self.upper = True
        self.transpose_a = True
        self.unit_diagonal = False

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


if __name__ == "__main__":
    unittest.main()
