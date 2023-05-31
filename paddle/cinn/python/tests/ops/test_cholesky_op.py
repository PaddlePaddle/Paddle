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
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestCholeskyOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        matrix = self.random([3, 3], "float32")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        self.inputs = {"x": x}
        self.upper = False

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


class TestCholeskyCase1(TestCholeskyOp):
    def init_case(self):
        matrix = self.random([5, 5], "float64")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        self.inputs = {"x": x}
        self.upper = True


class TestCholeskyCase2(TestCholeskyOp):
    def init_case(self):
        matrix = self.random([3, 3], "float32")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        x = x * np.ones(shape=(3, 3, 3))
        self.inputs = {"x": x}
        self.upper = False


class TestCholeskyCase3(TestCholeskyOp):
    def init_case(self):
        matrix = self.random([3, 3], "float64")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        x = x * np.ones(shape=(2, 3, 3, 3))
        self.inputs = {"x": x}
        self.upper = True


if __name__ == "__main__":
    unittest.main()
