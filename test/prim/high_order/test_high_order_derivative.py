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

sys.path.append("../../legacy_test")

import autodiff_checker_helper as ad_checker
import parameterized as param

import paddle
from paddle.base import core


class TestMulHigherOrderAD(unittest.TestCase):
    @param.parameterized.expand(
        [
            (
                paddle.multiply,
                [2, 3, 2],
                [2, 3, 2],
                'float32',
                2,
                ('cpu', 'gpu'),
            ),
            (
                paddle.multiply,
                [2, 3, 2],
                [2, 3, 2],
                'float64',
                2,
                ('cpu', 'gpu'),
            ),
            (
                paddle.multiply,
                [2, 3, 2, 1],
                [2, 3, 2, 4],
                'float64',
                2,
                ('cpu', 'gpu'),
            ),
        ]
    )
    def test_high_order_autodiff(
        self, func, shape1, shape2, dtype, order, places
    ):
        var_1 = np.random.randn(*shape1).astype(dtype)
        var_2 = np.random.randn(*shape2).astype(dtype)
        for place in places:
            if place == 'gpu' and not core.is_compiled_with_cuda():
                continue
            var1 = paddle.to_tensor(var_1, place=place)
            var2 = paddle.to_tensor(var_2, place=place)
            var1, var2 = paddle.broadcast_tensors(input=[var1, var2])
            ad_checker.check_vjp(
                func, [var1, var2], argnums=(0, 1), order=order
            )


class TestSinHigherOrderAD(unittest.TestCase):
    @param.parameterized.expand(
        [
            (paddle.sin, [2, 3, 2], 'float32', 4, ('cpu', 'gpu')),
            (paddle.sin, [2, 3, 2, 4], 'float64', 4, ('cpu', 'gpu')),
        ]
    )
    def test_high_order_autodiff(self, func, shape, dtype, order, places):
        var_1 = np.random.randn(*shape).astype(dtype)
        for place in places:
            if place == 'gpu' and not core.is_compiled_with_cuda():
                continue
            var1 = paddle.to_tensor(var_1, place=place)
            ad_checker.check_vjp(func, [var1], argnums=(0), order=order)


class TestCosHigherOrderAD(TestSinHigherOrderAD):
    @param.parameterized.expand(
        [
            (paddle.cos, [2, 3, 2], 'float32', 4, ('cpu', 'gpu')),
            (paddle.cos, [2, 3, 2, 4], 'float64', 4, ('cpu', 'gpu')),
        ]
    )
    def test_high_order_autodiff(self, func, shape, dtype, order, places):
        var_1 = np.random.randn(*shape).astype(dtype)
        for place in places:
            if place == 'gpu' and not core.is_compiled_with_cuda():
                continue
            var1 = paddle.to_tensor(var_1, place=place)
            ad_checker.check_vjp(func, [var1], argnums=(0), order=order)


class TestTanhHigherOrderAD(TestSinHigherOrderAD):
    @param.parameterized.expand(
        [
            (paddle.tanh, [2, 3, 2], 'float32', 4, ('cpu', 'gpu')),
            (paddle.tanh, [2, 3, 2, 4], 'float64', 4, ('cpu', 'gpu')),
        ]
    )
    def test_high_order_autodiff(self, func, shape, dtype, order, places):
        var_1 = np.random.randn(*shape).astype(dtype)
        for place in places:
            if place == 'gpu' and not core.is_compiled_with_cuda():
                continue
            var1 = paddle.to_tensor(var_1, place=place)
            ad_checker.check_vjp(func, [var1], argnums=(0), order=order)


if __name__ == "__main__":
    unittest.main()
