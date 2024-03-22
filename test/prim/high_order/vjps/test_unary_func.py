# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

sys.path.append("../../../legacy_test")

import autograd_checker_helper as grad_checker
import parameterized as param

import paddle
from paddle import base
from paddle.base import core


class TestUnaryHighGradCheck(unittest.TestCase):
    def func_wrapper(self, x):
        raise NotImplementedError("you must implement func_wrapper")

    def check_orders(self):
        return [2, 3, 4]

    def check_vjp(self, place):
        for order in self.check_orders():
            var = paddle.to_tensor(self.input, place=place)
            grad_checker.check_vjp(self.func_wrapper, [var], order)

    def check_high_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.check_vjp(p)


@param.parameterized_class(
    ('shape', "dtype"),
    [
        (
            [2, 3, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            "float32",
        ),
        (
            [3, 1, 1],
            "float64",
        ),
        (
            [2, 3, 1, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            "float64",
        ),
    ],
)
class TestSinHighGradCheck(TestUnaryHighGradCheck):
    @classmethod
    def setUpClass(cls):
        cls.shape = cls.shape
        cls.dtype = cls.dtype
        cls.input = np.random.randn(*cls.shape).astype(cls.dtype)

    def func_wrapper(self, x):
        return paddle.sin(x[0])

    def test_sin_high_grad(self):
        self.check_high_grad()


@param.parameterized_class(
    ('shape', "dtype", "orders"),
    [
        (
            [2, 3, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            "float32",
        ),
        (
            [3, 1, 1],
            "float64",
        ),
        (
            [2, 3, 1, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            "float64",
        ),
    ],
)
class TestCosHighGradCheck(TestUnaryHighGradCheck):
    @classmethod
    def setUpClass(cls):
        cls.shape = cls.shape
        cls.dtype = cls.dtype
        cls.input = np.random.randn(*cls.shape).astype(cls.dtype)

    def func_wrapper(self, x):
        return paddle.cos(x[0])

    def test_cos_high_grad(self):
        self.check_high_grad()


if __name__ == "__main__":
    unittest.main()
