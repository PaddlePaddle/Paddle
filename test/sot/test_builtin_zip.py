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

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle
from paddle.jit.sot import psdb, symbolic_translate
from paddle.jit.sot.utils import min_graph_size_guard, strict_mode_guard


def test_zip_1(x: int, y: int):
    for (id, val), val_ in zip(enumerate(range(x)), range(x)):
        if id % 2 == 0:
            y += val_
    return y


def test_zip_2(x: list):
    return list(zip(x, range(len(x))))


def test_zip_3(x: list):
    return tuple(zip(x, range(len(x))))


def test_zip_4(x: paddle.Tensor):
    sum = 0
    for idx, val in zip(range(len(x)), x):
        sum += val
    return sum


def test_zip_5(x: paddle.Tensor):
    sum = 0

    for idx, val in zip(range(len(x)), x):
        for i in range(idx):
            sum += val
    return sum


def test_zip_6(x: paddle.Tensor):
    sum = 0
    x = x.flatten()
    for idx, val in zip(range(len(x)), x):
        sum += val
    return sum


def test_zip_7(layer_list, x):
    sum = 0
    for idx, layer in zip(range(len(layer_list)), layer_list):
        sum += layer(x)
    return sum


def test_zip_8(iter_1, iter_2):
    sum = 0
    for a, b in zip(iter_1, iter_2):
        psdb.breakgraph()
        sum += a
        sum += b
    return sum


class TestZip(TestCaseBase):
    @test_with_faster_guard
    def test_simple_cases(self):
        x = 8
        y = 5
        ty = paddle.randn((10, 10))
        layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(3)]
        )
        self.assert_results(test_zip_1, x, y)
        self.assert_results(test_zip_2, [2, 4, 6, 8, 10])
        self.assert_results(test_zip_3, [2, 4, 6, 8, 10])
        self.assert_results(test_zip_4, ty)
        self.assert_results(test_zip_5, paddle.to_tensor([1, 2, 3]))
        self.assert_results(test_zip_6, ty)
        self.assert_results(test_zip_7, layer_list, paddle.randn((10,)))

    @test_with_faster_guard
    @min_graph_size_guard(0)
    def test_reconstruct(self):
        self.assert_results(test_zip_8, [1, 2, 3], [4, 5, 6])

    @test_with_faster_guard
    @strict_mode_guard(False)
    @min_graph_size_guard(0)
    def test_zip_user_defined_iter(self):
        sym_output = symbolic_translate(test_zip_8)(iter([1, 2, 3]), [4, 5, 6])
        paddle_output = test_zip_8(iter([1, 2, 3]), [4, 5, 6])
        self.assert_nest_match(sym_output, paddle_output)


if __name__ == "__main__":
    unittest.main()
