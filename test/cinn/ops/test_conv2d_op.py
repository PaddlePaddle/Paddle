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
from cinn.runtime import set_cinn_cudnn_deterministic
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle

set_cinn_cudnn_deterministic(True)
paddle.base.set_flags({'FLAGS_cudnn_deterministic': 1})


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestConv2dOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["dtype"]
        )
        self.w_np = self.random(
            shape=self.case["w_shape"], dtype=self.case["dtype"]
        )
        self.dy_np = self.random(
            shape=self.case["dy_shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = paddle.to_tensor(self.w_np, stop_gradient=False)
        y = paddle.nn.functional.conv2d(
            x,
            weight,
            stride=self.case["stride"],
            padding=self.case["padding"],
            dilation=self.case["dilation"],
            groups=self.case["groups"],
            data_format=self.case["data_format"],
        )
        self.paddle_outputs = [y]
        self.paddle_grads = self.get_paddle_grads(
            [y], [x, weight], [self.dy_np]
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["x_shape"], "x"
        )
        weight = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]),
            self.case["w_shape"],
            "weight",
        )
        dy = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]),
            self.case["dy_shape"],
            "dy",
        )

        if self.case["data_format"] == "NCHW":
            y = builder.conv2d(
                x,
                weight,
                strides=self.case["stride"],
                paddings=self.case["padding"],
                dilations=self.case["dilation"],
                groups=self.case["groups"],
                data_format=self.case["data_format"],
            )
            x_grad = builder.conv(
                weight,
                dy,
                data_format=self.case["data_format"],
                conv_type="backward_data",
                output_shape=x.shape(),
            )
            weight_grad = builder.conv(
                x,
                dy,
                data_format=self.case["data_format"],
                conv_type="backward_filter",
                output_shape=weight.shape(),
            )
        elif self.case["data_format"] == "NHWC":
            weight_t = builder.transpose(weight, [0, 2, 3, 1])
            y = builder.conv2d(
                x,
                weight_t,
                strides=self.case["stride"],
                paddings=self.case["padding"],
                dilations=self.case["dilation"],
                groups=self.case["groups"],
                data_format=self.case["data_format"],
            )
            x_grad = builder.conv(
                weight_t,
                dy,
                data_format=self.case["data_format"],
                conv_type="backward_data",
                output_shape=x.shape(),
            )
            w_grad = builder.conv(
                x,
                dy,
                data_format=self.case["data_format"],
                conv_type="backward_filter",
                output_shape=weight_t.shape(),
            )
            weight_grad = builder.transpose(w_grad, [0, 3, 1, 2])

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, weight, dy],
            [self.x_np, self.w_np, self.dy_np],
            [y, x_grad, weight_grad],
            passes=[],
        )

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestConv2dOpAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConv2dCase"
        self.cls = TestConv2dOp
        self.inputs = [
            {
                "x_shape": [3, 16, 32, 32],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 16, 30, 30],
                "data_format": "NCHW",
            },
            {
                "x_shape": [3, 16, 64, 64],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 16, 62, 62],
                "data_format": "NCHW",
            },
            {
                "x_shape": [3, 32, 32, 16],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 30, 30, 16],
                "data_format": "NHWC",
            },
            {
                "x_shape": [3, 64, 64, 16],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 62, 62, 16],
                "data_format": "NHWC",
            },
        ]
        self.dtypes = [
            {
                "dtype": "float16",
                "max_relative_error": 1e-3,
            },
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "groups": 1,
            },
        ]


# Cause Conv2d backward_fliter mode do not support NHWC
class TestConv2dOpFP64(TestConv2dOpAll):
    def init_attrs(self):
        super().init_attrs()
        self.inputs = [
            {
                "x_shape": [3, 16, 32, 32],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 16, 30, 30],
                "data_format": "NCHW",
            },
            {
                "x_shape": [3, 16, 64, 64],
                "w_shape": [16, 16, 3, 3],
                "dy_shape": [3, 16, 62, 62],
                "data_format": "NCHW",
            },
        ]
        self.dtypes = [
            {
                "dtype": "float64",
            },
        ]


if __name__ == "__main__":
    TestConv2dOpAll().run()
    TestConv2dOpFP64().run()
