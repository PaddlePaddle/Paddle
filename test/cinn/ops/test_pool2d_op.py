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

from cinn.common import is_compiled_with_cudnn
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle
from paddle import _C_ops


@OpTestTool.skip_if(
    not is_compiled_with_cudnn(), "x86 test will be skipped due to timeout."
)
class TestPool2dOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = _C_ops.pool2d(
            x,
            self.case["kernel_size"],
            self.case["stride"],
            self.case["padding"],
            self.case["ceil_mode"],
            self.case["exclusive"],
            self.case["data_format"],
            self.case["pooling_type"],
            self.case["global_pooling"],
            self.case["adaptive"],
            self.case["padding_algorithm"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x"
        )
        out = builder.pool2d(
            x,
            pooling_type=self.case["pooling_type"],
            kernel_size=self.case["kernel_size"],
            stride=self.case["stride"],
            padding=self.case["padding"],
            ceil_mode=self.case["ceil_mode"],
            exclusive=self.case["exclusive"],
            data_format=self.case["data_format"],
            global_pooling=self.case["global_pooling"],
            adaptive=self.case["adaptive"],
            padding_algorithm=self.case["padding_algorithm"],
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.x_np], [out], passes=[]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestPool2dOpAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestPool2dOpCase"
        self.cls = TestPool2dOp
        self.inputs = [
            {
                "shape": [1, 3, 32, 32],
                "data_format": "NCHW",
            },
            {
                "shape": [32, 3, 64, 64],
                "data_format": "NCHW",
            },
            {
                "shape": [1, 32, 32, 3],
                "data_format": "NHWC",
            },
            {
                "shape": [32, 64, 64, 3],
                "data_format": "NHWC",
            },
        ]
        self.dtypes = [
            {
                "dtype": "float16",
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
                "pooling_type": "max",
                "kernel_size": [2, 2],
                "stride": [1, 1],
                "padding": [0, 0],
                "padding_algorithm": "VALID",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "max",
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "padding_algorithm": "SAME",
                "global_pooling": False,
                "ceil_mode": True,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "max",
                "kernel_size": [5, 5],
                "stride": [3, 3],
                "padding": [2, 2],
                "padding_algorithm": "EXPLICIT",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": False,
                "adaptive": False,
            },
            {
                "pooling_type": "avg",
                "kernel_size": [2, 2],
                "stride": [1, 1],
                "padding": [0, 0],
                "padding_algorithm": "VALID",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "avg",
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "padding_algorithm": "SAME",
                "global_pooling": True,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "avg",
                "kernel_size": [5, 5],
                "stride": [3, 3],
                "padding": [1, 1],
                "padding_algorithm": "EXPLICIT",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": True,
            },
        ]


@OpTestTool.skip_if(
    not is_compiled_with_cudnn(), "x86 test will be skipped due to timeout."
)
class TestPool2dBackwardOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )
        self.dy_np = self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        forward_out = _C_ops.pool2d(
            x,
            self.case["kernel_size"],
            self.case["stride"],
            self.case["padding"],
            self.case["ceil_mode"],
            self.case["exclusive"],
            self.case["data_format"],
            self.case["pooling_type"],
            self.case["global_pooling"],
            self.case["adaptive"],
            self.case["padding_algorithm"],
            True,  # Need in paddlepaddle-2.4.2, will be removed in paddlepaddle-2.5
        )
        self.paddle_outputs = [forward_out]
        self.paddle_grads = self.get_paddle_grads(
            [forward_out], [x], [self.dy_np]
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        # forward
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x"
        )
        y = builder.pool2d(
            x,
            kernel_size=self.case["kernel_size"],
            stride=self.case["stride"],
            padding=self.case["padding"],
            ceil_mode=self.case["ceil_mode"],
            exclusive=self.case["exclusive"],
            data_format=self.case["data_format"],
            pooling_type=self.case["pooling_type"],
            global_pooling=self.case["global_pooling"],
            adaptive=self.case["adaptive"],
            padding_algorithm=self.case["padding_algorithm"],
        )
        # backward
        dy = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "dy"
        )
        dx = builder.pool2d_grad(
            x,
            y,
            dy,
            kernel_size=self.case["kernel_size"],
            stride=self.case["stride"],
            padding=self.case["padding"],
            ceil_mode=self.case["ceil_mode"],
            exclusive=self.case["exclusive"],
            data_format=self.case["data_format"],
            pooling_type=self.case["pooling_type"],
            global_pooling=self.case["global_pooling"],
            adaptive=self.case["adaptive"],
            padding_algorithm=self.case["padding_algorithm"],
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, dy], [self.x_np, self.dy_np], [y, dx], passes=[]
        )
        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1]]

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestPool2dBackwardAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestPool2dBackwardCase"
        self.cls = TestPool2dBackwardOp
        self.inputs = [
            {
                "shape": [1, 3, 32, 32],
                "data_format": "NCHW",
            },
            {
                "shape": [1, 32, 32, 3],
                "data_format": "NHWC",
            },
        ]
        self.dtypes = [
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
                "pooling_type": "max",
                "kernel_size": [5, 5],
                "stride": [2, 2],
                "padding": [0, 0],
                "padding_algorithm": "SAME",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "max",
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "padding_algorithm": "VALID",
                "global_pooling": False,
                "ceil_mode": True,
                "exclusive": False,
                "adaptive": False,
            },
            {
                "pooling_type": "avg",
                "kernel_size": [2, 2],
                "stride": [2, 2],
                "padding": [0, 0],
                "padding_algorithm": "SAME",
                "global_pooling": True,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": False,
            },
            {
                "pooling_type": "avg",
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "padding_algorithm": "EXPLICIT",
                "global_pooling": False,
                "ceil_mode": False,
                "exclusive": True,
                "adaptive": True,
            },
        ]


if __name__ == "__main__":
    TestPool2dOpAll().run()
    TestPool2dBackwardAll().run()
