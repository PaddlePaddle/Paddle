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
from typing import TYPE_CHECKING

from test_case_base import TestCaseBase, test_with_faster_guard

from paddle.jit import sot
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard

if TYPE_CHECKING:
    from collections.abc import Iterable


def double_num(num: float):
    return num * 2


def double_num_with_breakgraph(num: float):
    sot.psdb.breakgraph()
    return num * 2


@check_no_breakgraph
def test_map_list(x: list):
    return list(map(double_num, x))


@check_no_breakgraph
def test_map_list_comprehension(x: list):
    return [i for i in map(double_num, x)]  # noqa: C416


@check_no_breakgraph
def test_map_tuple(x: tuple):
    return tuple(map(double_num, x))


@check_no_breakgraph
def test_map_tuple_comprehension(x: tuple):
    return [i for i in map(double_num, x)]  # noqa: C416


@check_no_breakgraph
def test_map_range(x: Iterable):
    return list(map(double_num, x))


@check_no_breakgraph
def test_map_range_comprehension(x: Iterable):
    return [i for i in map(double_num, x)]  # noqa: C416


def add_dict_prefix(key: str):
    return f"dict_{key}"


@check_no_breakgraph
def test_map_dict(x: dict):
    return list(map(add_dict_prefix, x))


@check_no_breakgraph
def test_map_dict_comprehension(x: dict):
    return [i for i in map(add_dict_prefix, x)]  # noqa: C416


def test_map_list_with_breakgraph(x: list):
    return list(map(double_num_with_breakgraph, x))


@check_no_breakgraph
def test_map_unpack(x: list):
    a, b, c, d = map(double_num, x)
    return a, b, c, d


@check_no_breakgraph
def test_map_for_loop(x: list):
    res = 0
    for i in map(double_num, x):
        res += i
    return res


class TestMap(TestCaseBase):
    @test_with_faster_guard
    def test_map(self):
        self.assert_results(test_map_list, [1, 2, 3, 4])
        self.assert_results(test_map_tuple, (1, 2, 3, 4))
        self.assert_results(test_map_range, range(5))
        self.assert_results(test_map_dict, {"a": 1, "b": 2, "c": 3})

    @test_with_faster_guard
    def test_map_comprehension(self):
        self.assert_results(test_map_list_comprehension, [1, 2, 3, 4])
        self.assert_results(test_map_tuple_comprehension, (1, 2, 3, 4))
        self.assert_results(test_map_range_comprehension, range(5))
        self.assert_results(
            test_map_dict_comprehension, {"a": 1, "b": 2, "c": 3}
        )

    def test_map_with_breakgraph(self):
        with strict_mode_guard(False):
            self.assert_results(test_map_list_with_breakgraph, [1, 2, 3, 4])

    @test_with_faster_guard
    def test_map_unpack(self):
        self.assert_results(test_map_unpack, [1, 2, 3, 4])

    @test_with_faster_guard
    def test_map_for_loop(self):
        self.assert_results(test_map_for_loop, [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
