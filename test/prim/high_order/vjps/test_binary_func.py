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


class TestBinaryHighGradCheck(unittest.TestCase):
    def func_wrapper(self, x):
        raise NotImplementedError("you must implement func_wrapper")

    def check_orders(self):
        return [2, 3, 4]

    def check_vjp(self, place):
        for order in self.check_orders():
            var_1 = paddle.to_tensor(self.input1, place=place)
            var_2 = paddle.to_tensor(self.input2, place=place)
            var_1, var_2 = paddle.broadcast_tensors(input=[var_1, var_2])
            grad_checker.check_vjp(self.func_wrapper, [var_1, var_2], order)

    def check_high_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.check_vjp(p)


@param.parameterized_class(
    ('shape1', 'shape2', "dtype"),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
            "float64",
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
            "float32",
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
            "float64",
        ),
    ],
)
class TestMulHighGradCheck(TestBinaryHighGradCheck):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2
        cls.dtype = cls.dtype
        cls.input1 = np.random.randn(*cls.shape1).astype(cls.dtype)
        cls.input2 = np.random.randn(*cls.shape2).astype(cls.dtype)

    def func_wrapper(self, x):
        return paddle.multiply(x[0], x[1])

    def check_orders(self):
        return [2]

    def test_multiply_high_grad(self):
        self.check_high_grad()


if __name__ == "__main__":
    unittest.main()
