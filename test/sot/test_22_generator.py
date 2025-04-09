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

import operator
import unittest
from functools import reduce

from test_case_base import TestCaseBase

from paddle.jit import sot
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils.envs import strict_mode_guard


def create_simple_generator(a, b):
    yield a
    yield b


@check_no_breakgraph
def simple_generator_user():
    gen = create_simple_generator(1, 2)
    x = next(gen)
    y = next(gen)
    return x, y


@check_no_breakgraph
def genexpr_user():
    gen = (i for i in range(2))
    x = next(gen)
    y = next(gen)
    return x, y


def echo():
    recv = None
    recv = yield recv
    recv = yield recv
    recv = yield recv


@check_no_breakgraph
def generator_send():
    s = echo()
    init = next(s)
    recv_1 = s.send(2)
    recv_2 = s.send(3)
    return init, recv_1, recv_2


def yield_from_generator():
    yield from create_simple_generator(1, 2)


@check_no_breakgraph
def generator_yield_from_generator_user():
    s = yield_from_generator()
    x = next(s)
    y = next(s)
    return x, y


def yield_from_iterable():
    yield from [1, 2]


def generator_yield_from_iterable_user():
    s = yield_from_iterable()
    x = next(s)
    y = next(s)
    return x, y


def for_iterate_generator():
    out = 0
    for i in create_simple_generator(1, 2):
        out += i
    return out


def create_simple_generator_with_breakgraph():
    yield 1
    sot.psdb.breakgraph()
    yield 2


def simple_generator_user_with_breakgraph():
    gen = create_simple_generator_with_breakgraph()
    x = next(gen)
    y = next(gen)
    return x + y


def create_simple_generator_with_outer_breakgraph():
    yield 1
    yield 2


def simple_generator_user_with_outer_breakgraph():
    gen = create_simple_generator_with_outer_breakgraph()
    x = next(gen)
    sot.psdb.breakgraph()
    y = next(gen)
    return x + y


class TestGeneratorCommon(TestCaseBase):
    def test_generator_simple(self):
        self.assert_results(simple_generator_user)

    def test_generator_send(self):
        self.assert_results(generator_send)

    def test_genexpr(self):
        self.assert_results(genexpr_user)

    def test_generator_yield_from_generator(self):
        self.assert_results(generator_yield_from_generator_user)

    def test_generator_yield_from_iterable(self):
        self.assert_results(generator_yield_from_iterable_user)

    def test_for_iterate_generator(self):
        self.assert_results(for_iterate_generator)

    @strict_mode_guard(False)
    def test_generator_simple_with_breakgraph(self):
        self.assert_results(simple_generator_user_with_breakgraph)

    @strict_mode_guard(False)
    def test_generator_simple_with_outer_breakgraph(self):
        self.assert_results(simple_generator_user_with_outer_breakgraph)


@check_no_breakgraph
def generator_sum():
    return sum(i for i in range(10))


@check_no_breakgraph
def generator_max():
    return max((-2) ** i for i in range(10))


@check_no_breakgraph
def generator_min():
    return min((-2) ** i for i in range(10))


@check_no_breakgraph
def generator_reduce():
    return reduce(operator.add, (i for i in range(10)))


@check_no_breakgraph
def generator_map():
    return list(map(lambda x: x**2, (i for i in range(10))))  # noqa: C417


class TestGeneratorDispatch(TestCaseBase):
    def test_generator_sum(self):
        self.assert_results(generator_sum)

    def test_generator_max(self):
        self.assert_results(generator_max)

    def test_generator_min(self):
        self.assert_results(generator_min)

    def test_generator_reduce(self):
        self.assert_results(generator_reduce)

    def test_generator_map(self):
        self.assert_results(generator_map)


if __name__ == "__main__":
    unittest.main()
