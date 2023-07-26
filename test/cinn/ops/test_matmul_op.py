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
class TestMatmulOp(OpTest):
    def setUp(self):
        # print(f'{self.__class__.__name__}: {self.case}')
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.case["x_shape"], self.case["dtype"])
        self.y_np = self.random(self.case["y_shape"], self.case["dtype"])

    def paddle_func(self, x, y):
        return paddle.matmul(
            x,
            y,
            transpose_x=self.case["transx"],
            transpose_y=self.case["transy"],
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        y = paddle.to_tensor(self.y_np, stop_gradient=True)
        out = self.paddle_func(x, y)
        self.paddle_outputs = [out]

    def cinn_func(self, builder, x, y):
        return builder.matmul(
            x,
            y,
            transpose_x=self.case["transx"],
            transpose_y=self.case["transy"],
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("matmul")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["x_shape"], "x"
        )
        y = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["y_shape"], "y"
        )
        out = self.cinn_func(builder, x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out], passes=[]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        # 1e-6 is same as the atol parameter of np.allclose
        max_absolute_error = (
            self.case["max_absolute_error"]
            if "max_absolute_error" in self.case
            else 1e-6
        )
        self.check_outputs_and_grads(
            max_relative_error=max_relative_error,
            max_absolute_error=max_absolute_error,
        )


class TestMatmulOpShapeDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMatmulOpCase"
        self.cls = TestMatmulOp
        self.inputs = [
            {
                "x_shape": [1024],
                "y_shape": [1024],
            },
            {
                "x_shape": [4, 4],
                "y_shape": [4, 4],
            },
            {
                "x_shape": [4, 16],
                "y_shape": [16, 32],
            },
            {
                "x_shape": [5, 4, 16],
                "y_shape": [5, 16, 32],
            },
            {
                # Matrix mul row vector
                "x_shape": [4, 16],
                "y_shape": [16],
            },
            {
                # Matrix mul col vector
                "x_shape": [4, 16],
                "y_shape": [16, 1],
            },
            {
                "x_shape": [8, 16, 4],
                "y_shape": [1, 4, 16],
            },
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            # {
            #     "dtype": "bfloat16",
            # },
            {
                "dtype": "float16",
                "max_relative_error": 1e-2,
                "max_absolute_error": 1e-2,
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
                "transx": False,
                "transy": False,
            },
        ]


class TestMatmulOpTrans(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMatmulOpCase"
        self.cls = TestMatmulOp
        self.inputs = [
            {
                "x_shape": [16, 4],
                "y_shape": [16, 32],
                "transx": True,
                "transy": False,
            },
            {
                "x_shape": [5, 16, 4],
                "y_shape": [5, 16, 32],
                "transx": True,
                "transy": False,
            },
            {
                "x_shape": [8, 4, 16],
                "y_shape": [4, 16],
                "transx": True,
                "transy": False,
            },
            {
                "x_shape": [4, 16],
                "y_shape": [32, 16],
                "transx": False,
                "transy": True,
            },
            {
                "x_shape": [10, 12, 128, 64],
                "y_shape": [10, 12, 128, 64],
                "transx": False,
                "transy": True,
            },
            {
                "x_shape": [16, 4],
                "y_shape": [32, 16],
                "transx": True,
                "transy": True,
            },
            {
                "x_shape": [10, 12, 64, 128],
                "y_shape": [10, 12, 128, 64],
                "transx": True,
                "transy": True,
            },
            {
                "x_shape": [128],
                "y_shape": [10, 12, 128, 64],
                "transx": True,
                "transy": False,
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = []


class TestMatmulTransposePass(TestMatmulOp):
    def paddle_func(self, x, y):
        out = paddle.matmul(
            x,
            y,
            transpose_x=self.case["transx"],
            transpose_y=self.case["transy"],
        )
        return paddle.transpose(out, self.case["perm"])

    def cinn_func(self, builder, x, y):
        out = builder.matmul(
            x,
            y,
            transpose_x=self.case["transx"],
            transpose_y=self.case["transy"],
        )
        return builder.transpose(out, self.case["perm"])


class TestMatmulTransposePassAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestMatmulTransposePassCase"
        self.cls = TestMatmulTransposePass
        self.inputs = [
            {
                "x_shape": [32, 64],
                "y_shape": [64, 128],
                "perm": [1, 0],
                "transx": False,
                "transy": False,
            },
            {
                "x_shape": [10, 1, 128, 64],
                "y_shape": [10, 12, 64, 128],
                "perm": [0, 1, 3, 2],
                "transx": False,
                "transy": False,
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestMatmulOpShapeDtype().run()
    TestMatmulOpTrans().run()
    TestMatmulTransposePassAll().run()
