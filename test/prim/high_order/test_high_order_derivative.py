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


class TestMulHigherOrderAD(unittest.TestCase):
    order = 2

    def wrap_func(self, args, kwargs):
        return paddle.multiply(args[0], args[1])

    @param.parameterized.expand(
        [
            ([2, 3, 2], [2, 3, 2], 'float32', 'cpu'),
            ([2, 3, 2], [2, 3, 2], 'float32', 'gpu'),
            ([2, 3, 2], [2, 3, 2], 'float32', 'cpu'),
            ([2, 3, 2, 1], [2, 3, 2, 4], 'float64', 'gpu'),
        ]
    )
    def test_high_order_autodiff(self, shape1, shape2, dtype, place):
        var_1 = np.random.randn(*shape1).astype(dtype)
        var_2 = np.random.randn(*shape2).astype(dtype)
        var1 = paddle.to_tensor(var_1, place=place)
        var2 = paddle.to_tensor(var_2, place=place)
        var1, var2 = paddle.broadcast_tensors(input=[var1, var2])
        for order in range(2, self.order + 1):
            ad_checker.check_vjp(
                self.wrap_func, [var1, var2], argnums=(0, 1), order=order
            )


class TestSinHigherOrderAD(unittest.TestCase):
    order = 4

    def wrap_func(self, args, kwargs):
        return paddle.sin(args[0])

    @param.parameterized.expand(
        [
            ([2, 3, 2], 'float32', 'cpu'),
            ([2, 3, 2], 'float32', 'gpu'),
            ([2, 3, 2], 'float32', 'cpu'),
            ([2, 3, 2, 4], 'float64', 'gpu'),
        ]
    )
    def test_high_order_autodiff(self, shape, dtype, place):
        var_1 = np.random.randn(*shape).astype(dtype)
        var1 = paddle.to_tensor(var_1, place=place)
        for order in range(2, self.order + 1):
            ad_checker.check_vjp(
                self.wrap_func, [var1], argnums=(0), order=order
            )


class TestCosHigherOrderAD(TestSinHigherOrderAD):
    def wrap_func(self, args, kwargs):
        return paddle.cos(args[0])


class TestTanhHigherOrderAD(TestSinHigherOrderAD):
    def wrap_func(self, args, kwargs):
        return paddle.tanh(args[0])


if __name__ == "__main__":
    unittest.main()
