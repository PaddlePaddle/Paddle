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
from op_mapper_test import OpMapperTest

import paddle


class TestUnaryOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}

    def set_op_type(self):
        return "sqrt"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqrtOp(TestUnaryOp):
    def set_op_type(self):
        return "sqrt"


class TestGeluOp(TestUnaryOp):
    def set_op_type(self):
        return "gelu"


class TestSigmoidOp(TestUnaryOp):
    def set_op_type(self):
        return "sigmoid"


class TestExpOp(TestUnaryOp):
    def set_op_type(self):
        return "exp"


class TestErfOp(TestUnaryOp):
    def set_op_type(self):
        return "erf"


class TestRsqrtOp(TestUnaryOp):
    def set_op_type(self):
        return "rsqrt"


class TestSinOp(TestUnaryOp):
    def set_op_type(self):
        return "sin"


class TestCosOp(TestUnaryOp):
    def set_op_type(self):
        return "cos"


class TestTanOp(TestUnaryOp):
    def set_op_type(self):
        return "tan"


class TestSinhOp(TestUnaryOp):
    def set_op_type(self):
        return "sinh"


class TestCoshOp(TestUnaryOp):
    def set_op_type(self):
        return "cosh"


class TestTanhOp(TestUnaryOp):
    def set_op_type(self):
        return "tanh"


class TestAsinOp(TestUnaryOp):
    def set_op_type(self):
        return "asin"


class TestAcosOp(TestUnaryOp):
    def set_op_type(self):
        return "acos"


class TestAtanOp(TestUnaryOp):
    def set_op_type(self):
        return "atan"


class TestAsinhOp(TestUnaryOp):
    def set_op_type(self):
        return "asinh"


class TestAcoshOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", 1.0, 10.0)}

    def set_op_type(self):
        return "acosh"


class TestAtanhOp(TestUnaryOp):
    def set_op_type(self):
        return "atanh"


class TestSignOp(TestUnaryOp):
    def set_op_type(self):
        return "sign"


class TestAbsOp(TestUnaryOp):
    def set_op_type(self):
        return "abs"


class TestReciprocalOp(TestUnaryOp):
    def set_op_type(self):
        return "reciprocal"


class TestFloorOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_op_type(self):
        return "floor"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestCeilOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_op_type(self):
        return "ceil"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestRoundOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_op_type(self):
        return "round"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTruncOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_op_type(self):
        return "trunc"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsNanOp(TestUnaryOp):
    def set_op_type(self):
        return "isnan_v2"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsNanCase1(TestIsNanOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.nan] * 64


class TestIsFiniteOp(TestUnaryOp):
    def set_op_type(self):
        return "isfinite_v2"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsFiniteCase1(TestIsFiniteOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestIsInfOp(TestUnaryOp):
    def set_op_type(self):
        return "isinf_v2"

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsInfCase1(TestIsInfOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


if __name__ == "__main__":
    unittest.main()
