# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@paddle.no_grad()
def inner_no_grad_fn(x, y):
    return x * y + x**2


@check_no_breakgraph
def no_grad_fn_caller(x, y):
    z = inner_no_grad_fn(x * y, y)
    a = x * y + x**3 - 1
    return z + a


@check_no_breakgraph
@paddle.no_grad()
def outer_no_grad_fn(x, y):
    z = x * y + x**2
    a = inner_no_grad_fn(x, y)
    return z + a


class TestNoGrad(TestCaseBase):
    def test_inner_no_grad(self):
        x = paddle.randn([10, 3])
        y = paddle.randn([10, 3])
        x.stop_gradient = False
        y.stop_gradient = False
        self.assert_results_with_grad(
            [x, y],
            no_grad_fn_caller,
            x,
            y,
        )

    def test_outer_no_grad(self):
        x = paddle.randn([1, 3])
        y = paddle.randn([1, 3])
        x.stop_gradient = False
        y.stop_gradient = False
        self.assert_results_with_grad(
            [x, y],
            outer_no_grad_fn,
            x,
            y,
        )


if __name__ == "__main__":
    unittest.main()
