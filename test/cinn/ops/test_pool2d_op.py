#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cudnn(), "x86 test will be skipped due to timeout."
)
class TestPool2dOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float32")}
        self.polling_type = "max"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)

        if self.polling_type == "max":
            out = paddle.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.paddings,
                ceil_mode=self.ceil_mode,
                data_format=self.data_format,
            )
        elif self.polling_type == "avg":
            out = paddle.nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.paddings,
                ceil_mode=self.ceil_mode,
                exclusive=self.exclusive,
                data_format=self.data_format,
            )

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.pool2d(
            x,
            polling_type=self.polling_type,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.paddings,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            data_format=self.data_format,
        )

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[]
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestMaxPool2dNHWC(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 32, 32, 3], "float32")}
        self.polling_type = "max"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestMaxPool2dFP16(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float16")}
        self.polling_type = "max"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestMaxPool2dFP16NHWC(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 32, 32, 3], "float16")}
        self.polling_type = "max"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2d(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float32")}
        self.polling_type = "avg"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dNHWC(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 32, 32, 3], "float32")}
        self.polling_type = "avg"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dFP16(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 3, 32, 32], "float16")}
        self.polling_type = "avg"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dFP16NHWC(TestPool2dOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 32, 32, 3], "float16")}
        self.polling_type = "avg"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


@OpTestTool.skip_if(
    not is_compiled_with_cudnn(), "x86 test will be skipped due to timeout."
)
class TestPool2dBackwardOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([1, 3, 32, 32], "float32"),
            "dy": self.random([1, 3, 16, 16], "float32"),
        }
        self.polling_type = "max"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)

        if self.polling_type == "max":
            out = paddle.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.paddings,
                ceil_mode=self.ceil_mode,
                data_format=self.data_format,
            )
        elif self.polling_type == "avg":
            out = paddle.nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.paddings,
                ceil_mode=self.ceil_mode,
                exclusive=self.exclusive,
                data_format=self.data_format,
            )

        self.paddle_outputs = [out.numpy()]
        self.paddle_grads = self.get_paddle_grads(
            [out], [x], [self.inputs["dy"]]
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        # froward
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.pool2d(
            x,
            polling_type=self.polling_type,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.paddings,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            data_format=self.data_format,
        )

        # backward
        dy = builder.create_input(
            self.nptype2cinntype(self.inputs["dy"].dtype),
            self.inputs["dy"].shape,
            "dy",
        )
        dx = builder.pool2d_grad(
            x,
            y,
            dy,
            polling_type=self.polling_type,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.paddings,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            data_format=self.data_format,
        )

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, dy],
            [self.inputs["x"], self.inputs["dy"]],
            [y, dx],
            passes=[],
        )

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestMaxPool2dBackwardNHWC(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 32, 32, 3], "float32"),
            "dy": self.random([1, 16, 16, 3], "float32"),
        }
        self.polling_type = "max"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestMaxPool2dBackwardFP16(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 3, 32, 32], "float16"),
            "dy": self.random([1, 3, 16, 16], "float16"),
        }
        self.polling_type = "max"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestMaxPool2dBackwardFP16NHWC(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 32, 32, 3], "float16"),
            "dy": self.random([1, 16, 16, 3], "float16"),
        }
        self.polling_type = "max"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dBackward(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 3, 32, 32], "float32"),
            "dy": self.random([1, 3, 16, 16], "float32"),
        }
        self.polling_type = "avg"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dBackwardNHWC(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 32, 32, 3], "float32"),
            "dy": self.random([1, 16, 16, 3], "float32"),
        }
        self.polling_type = "avg"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dBackwardFP16(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 3, 32, 32], "float16"),
            "dy": self.random([1, 3, 16, 16], "float16"),
        }
        self.polling_type = "avg"
        self.data_format = "NCHW"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


class TestAvgPool2dBackwardFP16NHWC(TestPool2dBackwardOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 32, 32, 3], "float16"),
            "dy": self.random([1, 16, 16, 3], "float16"),
        }
        self.polling_type = "avg"
        self.data_format = "NHWC"
        self.kernel_size = [2, 2]
        self.stride = [2, 2]
        self.paddings = [0, 0]
        self.ceil_mode = False
        self.exclusive = True


if __name__ == "__main__":
    unittest.main()
