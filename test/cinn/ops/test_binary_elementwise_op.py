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
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestBinaryOp(OpTest):
    def setUp(self):
        self.init_case()

    def get_x_data(self):
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_axis_value(self):
        return -1

    def init_case(self):
        self.inputs = {"x": self.get_x_data(), "y": self.get_y_data()}
        self.axis = self.get_axis_value()

    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.add(x, y, axis)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.",
            )
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = (
                np.arange(0, axis).tolist()
                + np.arange(axis + y_rank, x_rank).tolist()
            )

            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(self.inputs["x"].shape), len(self.inputs["y"].shape), self.axis
        )
        y_t = (
            paddle.unsqueeze(y, axis=unsqueeze_axis)
            if len(unsqueeze_axis) > 0
            else y
        )
        out = self.paddle_func(x, y_t)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("binary_elementwise_test")
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
        out = self.cinn_func(builder, x, y, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestAddOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.add(x, y, axis)


class TestAddOpFP64(TestAddOp):
    def get_x_data(self):
        return self.random([32, 64], 'float64', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'float64', -10.0, 10.0)


class TestAddOpFP16(TestAddOp):
    def get_x_data(self):
        return self.random([32, 64], 'float16', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'float16', -10.0, 10.0)


class TestAddOpInt32(TestAddOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'int32', -10.0, 10.0)


class TestAddOpInt64(TestAddOp):
    def get_x_data(self):
        return self.random([32, 64], 'int64', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'int64', -10.0, 10.0)


class TestSubtractOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.subtract(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.subtract(x, y, axis)


class TestDivideOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.divide(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.divide(x, y, axis)


class TestMultiplyOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.multiply(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.multiply(x, y, axis)


class TestFloorDivideOp(TestBinaryOp):
    def get_x_data(self):
        # avoid random generate 0
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )

    def get_y_data(self):
        # avoid random generate 0
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )

    def paddle_func(self, x, y):
        return paddle.floor_divide(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.floor_divide(x, y, axis)


class TestModOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.mod(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.mod(x, y, axis)


class TestModCase1(TestModOp):
    def get_x_data(self):
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )

    def get_y_data(self):
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )


class TestRemainderOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.remainder(x, y)

    def cinn_func(self, builder, x, y, axis):
        # paddle.remainder actual invoke mod function
        return builder.mod(x, y, axis)


class TestRemainderCase1(TestRemainderOp):
    def get_x_data(self):
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )

    def get_y_data(self):
        return (
            self.random([32, 64], 'int32', 1, 100)
            * np.random.choice([-1, 1], [1])[0]
        )


class TestMaxOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.maximum(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.max(x, y, axis)


class TestMinOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.minimum(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.min(x, y, axis)


class TestLogicalAndOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        return paddle.logical_and(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.logical_and(x, y, axis)


class TestLogicalOrOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        return paddle.logical_or(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.logical_or(x, y, axis)


class TestLogicalXorOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'bool')

    def get_y_data(self):
        return self.random([32, 64], 'bool')

    def paddle_func(self, x, y):
        return paddle.logical_xor(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.logical_xor(x, y, axis)


class TestBitwiseAndOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_and(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_and(x, y, axis)


class TestBitwiseOrOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_or(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_or(x, y, axis)


class TestBitwiseXorOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_xor(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_xor(x, y, axis)


class TestEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.equal(x, y, axis)


class TestNotEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.not_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.not_equal(x, y, axis)


class TestGreaterThanOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.greater_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.greater_than(x, y, axis)


class TestLessThanOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.less_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.less_than(x, y, axis)


class TestGreaterEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.greater_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.greater_equal(x, y, axis)


class TestLessEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.less_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.less_equal(x, y, axis)


class TestAtan2Op(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.atan2(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.atan2(x, y, axis)


if __name__ == "__main__":
    unittest.main()
