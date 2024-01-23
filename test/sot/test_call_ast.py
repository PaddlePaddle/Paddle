# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.jit.sot.utils import try_ast_func, with_control_flow_guard


@try_ast_func
def calc(x, y, z):
    if x < 5:
        a = x + y
        b = y - z
        c = a * b
        return c
    else:
        a = x - y
        b = y + z
        c = a * b
        return c


def inline_call_ast(x, y):
    a = x - y + 3
    b = x + y
    c = x * y
    z = calc(a, b, c)
    return z + a


class TestNumpyAdd(TestCaseBase):
    @with_control_flow_guard(True)
    def test_full_graph_ast(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        z = paddle.to_tensor([4])
        self.assert_results(calc, x, y, z)

    @with_control_flow_guard(True)
    def test_inline_ast(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(inline_call_ast, x, y)


if __name__ == "__main__":
    unittest.main()
