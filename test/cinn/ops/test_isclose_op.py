#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
class TestIsCloseOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        if self.case["nan_as_input"]:
            self.x_np = np.full(shape=self.case["shape"], fill_value=np.nan)
        else:
            self.x_np = self.random(
                shape=self.case["shape"], dtype=self.case["dtype"]
            )
        self.y_np = self.x_np + self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        shape = paddle.broadcast_shape(x.shape, y.shape)
        x = paddle.broadcast_to(x, shape)
        y = paddle.broadcast_to(y, shape)
        out = paddle.isclose(
            x, y, self.case["rtol"], self.case["atol"], self.case["equal_nan"]
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("isclose")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        y = builder.create_input(
            self.nptype2cinntype(self.y_np.dtype), self.y_np.shape, "y"
        )
        out = builder.isclose(
            x, y, self.case["rtol"], self.case["atol"], self.case["equal_nan"]
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsCloseShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsCloseOpCase"
        self.cls = TestIsCloseOp
        self.inputs = [
            {
                "shape": [1],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [512, 256],
            },
            {
                "shape": [128, 64, 32],
            },
            {
                "shape": [16, 8, 4, 2],
            },
            {
                "shape": [16, 8, 4, 2, 1],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "rtol": 1e-5,
                "atol": 1e-8,
                "equal_nan": False,
                "nan_as_input": False,
            },
        ]


class TestIsCloseDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsCloseOpCase"
        self.cls = TestIsCloseOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
        ]
        self.attrs = [
            {
                "rtol": 1e-5,
                "atol": 1e-8,
                "equal_nan": False,
                "nan_as_input": False,
            },
        ]


class TestIsCloseAttr(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsCloseOpCase"
        self.cls = TestIsCloseOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "rtol": 1e-3,
                "atol": 1e-3,
                "equal_nan": False,
                "nan_as_input": False,
            },
            {
                "rtol": 1e-5,
                "atol": 1e-5,
                "equal_nan": False,
                "nan_as_input": False,
            },
            {
                "rtol": 1e-8,
                "atol": 1e-8,
                "equal_nan": False,
                "nan_as_input": False,
            },
            {
                "rtol": 1e-5,
                "atol": 1e-8,
                "equal_nan": True,
                "nan_as_input": False,
            },
        ]


class TestIsCloseNAN(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestIsCloseOpCase"
        self.cls = TestIsCloseOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float64",
            },
        ]
        self.attrs = [
            {
                "rtol": 1e-5,
                "atol": 1e-8,
                "equal_nan": True,
                "nan_as_input": True,
            },
        ]


if __name__ == "__main__":
    TestIsCloseShape().run()
    TestIsCloseDtype().run()
    TestIsCloseAttr().run()
    TestIsCloseNAN().run()
