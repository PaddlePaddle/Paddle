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

# New Supported Instructions:
# UNPACK_SEQUENCE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle


def unpack_tuple(x: tuple[int, paddle.Tensor]):
    y, z = x
    return z + 1


def unpack_tensor(x: paddle.Tensor):
    a, b = x
    return (a, b)


def unpack_ex_tuple(x: tuple[int, int, paddle.Tensor]):
    *y, z = x
    return z + 1


def unpack_ex_tensor(x: paddle.Tensor):
    a, b, *c = x
    return (a, b)


def unpack_ex_tensor_2(x: paddle.Tensor):
    a, *b, c, d = x
    return (a, c)


class TestUnpack(TestCaseBase):
    def test_unpack_tuple(self):
        self.assert_results(unpack_tuple, (1, paddle.to_tensor(2)))

    def test_unpack_tensor(self):
        self.assert_results(unpack_tensor, paddle.to_tensor([2, 3]))

    @test_with_faster_guard
    def test_unpack_ex_tuple(self):
        self.assert_results(unpack_ex_tuple, (1, 1, paddle.to_tensor(2)))

    def test_unpack_ex_tensor(self):
        self.assert_results(unpack_ex_tensor, paddle.to_tensor([2, 3, 3, 3]))

    def test_unpack_ex_tensor_2(self):
        self.assert_results(unpack_ex_tensor_2, paddle.to_tensor([2, 3, 3, 3]))


if __name__ == "__main__":
    unittest.main()
