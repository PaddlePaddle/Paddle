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
from enum import IntEnum
from typing import Any

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

from paddle.jit.sot.psdb import check_no_breakgraph


class Direction(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Direction2(IntEnum):
    EAST = 1
    WEST = 2
    NORTH = 3
    SOUTH = 4


@check_no_breakgraph
def eq_func(x: IntEnum, y: IntEnum):
    return x == y


def eq_func_2(x: IntEnum, y: Any):
    return x == y


@check_no_breakgraph
def not_eq_func(x: IntEnum, y: IntEnum):
    return x != y


def not_eq_func_2(x: IntEnum, y: Any):
    return x != y


@check_no_breakgraph
def name_func(x: IntEnum):
    return x.name


@check_no_breakgraph
def is_func(x: IntEnum, y: IntEnum):
    return x is y


def get_func(x: int):
    return Direction(x)


@check_no_breakgraph
def get_func_2(x: str):
    return Direction[x]


class TestEnumMethod(TestCaseBase):
    def test_guard(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(eq_func, Direction.UP, Direction.UP)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(eq_func, Direction.UP, Direction.DOWN)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(eq_func, Direction.UP, Direction.UP)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(eq_func, Direction.UP, Direction2.EAST)
            self.assertEqual(ctx.translate_count, 3)

    def test_eq(self):
        self.assert_results(eq_func, Direction.UP, Direction.UP)
        self.assert_results(eq_func, Direction.RIGHT, Direction.LEFT)
        self.assert_results(eq_func, Direction.UP, Direction2.EAST)
        self.assert_results(eq_func_2, Direction.UP, 1)

    def test_not_eq(self):
        self.assert_results(not_eq_func, Direction.UP, Direction.UP)
        self.assert_results(not_eq_func, Direction.UP, Direction.DOWN)
        self.assert_results(not_eq_func, Direction.UP, Direction2.EAST)
        self.assert_results(not_eq_func_2, Direction.UP, 0)

    def test_name(self):
        self.assert_results(name_func, Direction.UP)

    def test_is(self):
        self.assert_results(is_func, Direction.UP, Direction.UP)
        self.assert_results(is_func, Direction.RIGHT, Direction.LEFT)
        self.assert_results(is_func, Direction.UP, Direction2.EAST)

    def test_get(self):
        # TODO(wangmingkai): implement “_reconstruct” for EnumVariable
        # self.assert_results(get_func, 1)
        self.assert_results(get_func_2, 'UP')


if __name__ == "__main__":
    unittest.main()
