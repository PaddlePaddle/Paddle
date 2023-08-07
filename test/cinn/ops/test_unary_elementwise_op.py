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

import unittest

import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestUnaryOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -10.0, 10.0)}

    def paddle_func(self, x):
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        return builder.abs(x)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = self.paddle_func(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("unary_elementwise_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqrtOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 1000.0)}

    def paddle_func(self, x):
        return paddle.sqrt(x)

    def cinn_func(self, builder, x):
        return builder.sqrt(x)


class TestSqrtOpFP64(TestSqrtOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float64', 1.0, 1000.0)}


class TestReluOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.nn.functional.relu(x)

    def cinn_func(self, builder, x):
        return builder.relu(x)


class TestSigmoidOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.nn.functional.sigmoid(x)

    def cinn_func(self, builder, x):
        return builder.sigmoid(x)


class TestIdentityOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.assign(x)

    def cinn_func(self, builder, x):
        return builder.identity(x)


class TestExpOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.exp(x)

    def cinn_func(self, builder, x):
        return builder.exp(x)


class TestExpOpFP64(TestExpOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float64', -10.0, 10.0)}


class TestErfOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.erf(x)

    def cinn_func(self, builder, x):
        return builder.erf(x)


class TestRsqrtOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 0.00001, 1.0)}

    def paddle_func(self, x):
        return paddle.rsqrt(x)

    def cinn_func(self, builder, x):
        return builder.rsqrt(x)


class TestLogOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        return paddle.log(x)

    def cinn_func(self, builder, x):
        return builder.log(x)


class TestLog2Op(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        return paddle.log2(x)

    def cinn_func(self, builder, x):
        return builder.log2(x)


class TestLog10Op(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

    def paddle_func(self, x):
        return paddle.log10(x)

    def cinn_func(self, builder, x):
        return builder.log10(x)


class TestFloorOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.floor(x)

    def cinn_func(self, builder, x):
        return builder.floor(x)


class TestCeilOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.ceil(x)

    def cinn_func(self, builder, x):
        return builder.ceil(x)


class TestRoundOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.round(x)

    def cinn_func(self, builder, x):
        return builder.round(x)


class TestTruncOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.trunc(x)

    def cinn_func(self, builder, x):
        return builder.trunc(x)


class TestSinOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sin(x)

    def cinn_func(self, builder, x):
        return builder.sin(x)


class TestCosOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.cos(x)

    def cinn_func(self, builder, x):
        return builder.cos(x)


class TestTanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.tan(x)

    def cinn_func(self, builder, x):
        return builder.tan(x)


class TestSinhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sinh(x)

    def cinn_func(self, builder, x):
        return builder.sinh(x)


class TestCoshOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.cosh(x)

    def cinn_func(self, builder, x):
        return builder.cosh(x)


class TestTanhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.tanh(x)

    def cinn_func(self, builder, x):
        return builder.tanh(x)


class TestAsinOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        return paddle.asin(x)

    def cinn_func(self, builder, x):
        return builder.asin(x)


class TestAcosOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        return paddle.acos(x)

    def cinn_func(self, builder, x):
        return builder.acos(x)


class TestAtanOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        return paddle.atan(x)

    def cinn_func(self, builder, x):
        return builder.atan(x)


class TestAsinhOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        return paddle.asinh(x)

    def cinn_func(self, builder, x):
        return builder.asinh(x)


class TestAcoshOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 100.0)}

    def paddle_func(self, x):
        return paddle.acosh(x)

    def cinn_func(self, builder, x):
        return builder.acosh(x)


class TestAtanhOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'float32', -1.0, 1.0)}

    def paddle_func(self, x):
        return paddle.atanh(x)

    def cinn_func(self, builder, x):
        return builder.atanh(x)


class TestLogicalNotOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'bool')}

    def paddle_func(self, x):
        return paddle.logical_not(x)

    def cinn_func(self, builder, x):
        return builder.logical_not(x)


class TestBitwiseNotOp(TestUnaryOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], 'int32', 1, 10000)}

    def paddle_func(self, x):
        return paddle.bitwise_not(x)

    def cinn_func(self, builder, x):
        return builder.bitwise_not(x)


class TestSignOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sign(x)

    def cinn_func(self, builder, x):
        return builder.sign(x)


class TestAbsOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        return builder.abs(x)


class TestIsNanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isnan(x)

    def cinn_func(self, builder, x):
        return builder.is_nan(x)


class TestIsNanCase1(TestIsNanOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.nan] * 64


class TestIsFiniteOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isfinite(x)

    def cinn_func(self, builder, x):
        return builder.is_finite(x)


class TestIsFiniteCase1(TestIsFiniteOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestIsInfOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isinf(x)

    def cinn_func(self, builder, x):
        return builder.is_inf(x)


class TestIsInfCase1(TestIsInfOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestNegOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.neg(x)

    def cinn_func(self, builder, x):
        return builder.negative(x)


class TestNegCase1(TestNegOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], low=-1.0, high=1.0)}


if __name__ == "__main__":
    unittest.main()
