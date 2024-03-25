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


def rot_two_return_a(a: paddle.Tensor, b: paddle.Tensor):
    b, a = a, b
    return a + 1


def rot_two_return_b(a: paddle.Tensor, b: paddle.Tensor):
    b, a = a, b
    return b + 2


def rot_three_return_a(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return a + 1


def rot_three_return_b(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return b + 1


def rot_three_return_c(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return c + 1


def rot_four_return_a(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return a + 1


def rot_four_return_b(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return b + 1


def rot_four_return_c(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return c + 1


def rot_four_return_d(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return d + 1


class TestRot(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)
        self.assert_results(rot_two_return_a, a, b)
        self.assert_results(rot_two_return_b, a, b)

        self.assert_results(rot_three_return_a, a, b, c)
        self.assert_results(rot_three_return_b, a, b, c)
        self.assert_results(rot_three_return_c, a, b, c)

        self.assert_results(rot_four_return_a, a, b, c, d)
        self.assert_results(rot_four_return_b, a, b, c, d)
        self.assert_results(rot_four_return_c, a, b, c, d)
        self.assert_results(rot_four_return_d, a, b, c, d)


if __name__ == "__main__":
    unittest.main()
