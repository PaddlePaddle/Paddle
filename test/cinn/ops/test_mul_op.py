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


def infer_shape(
    x_shape: list,
    y_shape: list,
    x_num_col_dim: int,
    y_num_col_dim: int,
    is_infer: bool,
):
    def flatten_shape(shape: list, num_col_dim: int) -> list:
        if len(shape) <= 2:
            return shape
        else:
            new_shape = [1, 1]
            for i, x in enumerate(shape):
                if i < num_col_dim:
                    new_shape[0] *= x
                else:
                    new_shape[1] *= x
            return new_shape

    x_new_shape = flatten_shape(x_shape, x_num_col_dim)
    y_new_shape = flatten_shape(y_shape, y_num_col_dim)
    out_shape = []
    for i in range(x_num_col_dim):
        out_shape.append(x_shape[i])
    if is_infer:
        for i in range(y_num_col_dim):
            out_shape.append(y_shape[i])
    else:
        for i in range(y_num_col_dim, len(y_shape)):
            out_shape.append(y_shape[i])
    return x_new_shape, y_new_shape, out_shape


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestMulOp(OpTest):
    def setUp(self):
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["dtype"]
        )
        self.y_np = self.random(
            shape=self.case["y_shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        x_shape, y_shape, out_shape = infer_shape(
            x.shape,
            y.shape,
            self.case["x_num_col_dims"],
            self.case["y_num_col_dims"],
            self.case["is_infer"],
        )
        x = paddle.reshape(x, x_shape)
        y = paddle.reshape(y, y_shape)
        if self.case["is_infer"]:
            out = paddle.matmul(x, y, transpose_x=False, transpose_y=True)
        else:
            out = paddle.matmul(x, y)
        out = paddle.reshape(out, out_shape)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("mul")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["x_shape"], "x"
        )
        y = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["y_shape"], "y"
        )
        out = builder.mul(
            x,
            y,
            x_num_col_dims=self.case["x_num_col_dims"],
            y_num_col_dims=self.case["y_num_col_dims"],
            is_infer=self.case["is_infer"],
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestMulOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMulOpShape"
        self.cls = TestMulOp
        self.inputs = [
            {
                "x_shape": [1, 1],
                "y_shape": [1, 1],
                "x_num_col_dims": 1,
                "y_num_col_dims": 1,
            },
            {
                "x_shape": [32, 64],
                "y_shape": [64, 32],
                "x_num_col_dims": 1,
                "y_num_col_dims": 1,
            },
            {
                "x_shape": [2, 3, 4],
                "y_shape": [4, 3, 2],
                "x_num_col_dims": 1,
                "y_num_col_dims": 2,
            },
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [2, 4, 8, 16],
                "x_num_col_dims": 2,
                "y_num_col_dims": 2,
            },
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [1, 1, 1, 1],
                "x_num_col_dims": 2,
                "y_num_col_dims": 2,
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "is_infer": False,
            },
        ]


class TestMulOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMulOpDtype"
        self.cls = TestMulOp
        self.inputs = [
            {
                "x_shape": [32, 64],
                "y_shape": [64, 32],
                "x_num_col_dims": 1,
                "y_num_col_dims": 1,
            },
        ]
        self.dtypes = [
            # cublas bf16 gemm requires GPU compute capability >= 80
            # {
            #     "dtype": "bfloat16",
            #     "max_relative_error": 1e-3,
            # },
            {
                "dtype": "float16",
                "max_relative_error": 1e-2,
            },
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
        ]
        self.attrs = [
            {
                "is_infer": False,
            },
        ]


class TestMulOpAttr(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMulOpAttr"
        self.cls = TestMulOp
        self.inputs = [
            {
                "x_shape": [16, 8, 4, 2],
                "y_shape": [16, 8, 4, 2],
                "x_num_col_dims": 2,
                "y_num_col_dims": 2,
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "is_infer": True,
            },
        ]


if __name__ == "__main__":
    TestMulOpShape().run()
    TestMulOpDtype().run()
    TestMulOpAttr().run()
