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

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def pop_jump_if_false(x: bool, y: paddle.Tensor):
    if x:
        y += 1
    else:
        y -= 1
    return y


@check_no_breakgraph
def pop_jump_if_true(x: bool, y: bool, z: paddle.Tensor):
    return (x or y) and z


@check_no_breakgraph
def jump_if_false_or_pop(x: bool, y: paddle.Tensor):
    return x and (y + 1)


@check_no_breakgraph
def jump_if_true_or_pop(x: bool, y: paddle.Tensor):
    return x or (y + 1)


@check_no_breakgraph
def jump_absolute(x: int, y: paddle.Tensor):
    while x > 0:
        y += 1
        x -= 1
    return y


@check_no_breakgraph
def pop_jump_if_none(x: bool, y: paddle.Tensor):
    if x is not None:
        y += 1
    else:
        y -= 1
    return y


@check_no_breakgraph
def pop_jump_if_not_none(x: bool, y: paddle.Tensor):
    if x is None:
        y += 1
    else:
        y -= 1
    return y


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)

true_tensor = paddle.to_tensor(True)
false_tensor = paddle.to_tensor(False)


class TestJump(TestCaseBase):
    def test_simple(self):
        self.assert_results(jump_absolute, 5, a)

        self.assert_results(pop_jump_if_false, True, a)
        self.assert_results(pop_jump_if_false, False, a)
        self.assert_results(jump_if_false_or_pop, True, a)
        self.assert_results(jump_if_false_or_pop, False, a)
        self.assert_results(jump_if_true_or_pop, True, a)
        self.assert_results(jump_if_true_or_pop, False, a)
        self.assert_results(pop_jump_if_true, True, False, a)
        self.assert_results(pop_jump_if_true, False, False, a)

        self.assert_results(pop_jump_if_none, None, a)
        self.assert_results(pop_jump_if_none, True, a)
        self.assert_results(pop_jump_if_not_none, None, a)
        self.assert_results(pop_jump_if_not_none, True, a)

    def test_breakgraph(self):
        self.assert_results(pop_jump_if_false, true_tensor, a)
        self.assert_results(jump_if_false_or_pop, true_tensor, a)
        self.assert_results(jump_if_true_or_pop, false_tensor, a)
        self.assert_results(pop_jump_if_true, true_tensor, false_tensor, a)
        self.assert_results(jump_absolute, 5, a)
        self.assert_results(pop_jump_if_false, false_tensor, a)
        self.assert_results(jump_if_false_or_pop, false_tensor, a)
        self.assert_results(jump_if_true_or_pop, false_tensor, a)
        self.assert_results(pop_jump_if_true, true_tensor, false_tensor, a)

        self.assert_results(pop_jump_if_none, true_tensor, a)
        self.assert_results(pop_jump_if_not_none, true_tensor, a)


def new_var_in_if():
    x = paddle.to_tensor(1)
    if x > 0:
        y = 1
    return y


class TestCreateVarInIf(TestCaseBase):
    def test_case(self):
        self.assert_results(new_var_in_if)


if __name__ == "__main__":
    unittest.main()
