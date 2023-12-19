# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle


class TestInplaceAssign(Dy2StTestBase):
    @test_ast_only
    def test_case0(self):
        a = paddle.ones((1024, 2)) * 1
        b = paddle.ones((1024, 3)) * 2
        c = paddle.ones((1024, 4)) * 3
        a._inplace_assign(b)
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        b._inplace_assign(c)
        np.testing.assert_array_equal(b.numpy(), c.numpy())

    @test_ast_only
    def test_case1(self):
        def func(x):
            a = 1 * x
            b = 2 * x
            a._inplace_assign(b)
            return a

        x = paddle.ones((1,))
        a = paddle.randn((1,))
        x.stop_gradient = False
        a.stop_gradient = False
        y = func(x)
        y.mean().backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.array([2.0]))

    def test_case2(self):
        def func(a, x):
            x = 2 * x
            x[:] = a * 2.0
            return x

        def forward(a, x):
            output = paddle.jit.to_static(func)(a, x)
            x._inplace_assign(output)
            return x

        x = paddle.ones((1,))
        a = paddle.randn((1,))
        x.stop_gradient = False
        a.stop_gradient = False
        y = forward(a, x)
        y.mean().backward()
        np.testing.assert_array_equal(a.grad.numpy(), np.array([2.0]))


if __name__ == "__main__":
    unittest.main()
