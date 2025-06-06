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

from __future__ import annotations

import unittest
from collections import namedtuple

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot import psdb

Singleton = namedtuple("Singleton", ["a"])
Pair = namedtuple("Pair", ["a", "b"])
Triplet = namedtuple("Triple", ["a", "b", "c"])
Quartet = namedtuple("Quartet", ["a", "b", "c", "d"])
QuartetWithDefault = namedtuple(
    "QuartetWithDefault", ["a", "b", "c", "d"], defaults=[1, 2, 3]
)


@psdb.check_no_breakgraph
def create_namedtuple():
    x = Singleton(1)
    y = Singleton(2)
    z = Singleton(3)
    return x.a + y.a + z.a


@psdb.check_no_breakgraph
def create_namedtuple_with_tensor(x):
    y = Pair(x + 1, x + 2)
    z = Pair(x + 3, x + 4)
    return y.a + y.b + z.a + z.b


@psdb.check_no_breakgraph
def load_namedtuple(x, y):
    return x.a + y.a + y.b


@psdb.check_no_breakgraph
def create_namedtuple_with_default():
    u = QuartetWithDefault(1, 2, 3, 4)
    v = QuartetWithDefault(5, 6, 7)
    w = QuartetWithDefault(8, 9)
    x = QuartetWithDefault(10)
    return (
        (u.a + u.b + u.c + u.d)
        + (v.a + v.b + v.c + v.d)
        + (w.a + w.b + w.c + w.d)
        + (x.a + x.b + x.c + x.d)
    )


@psdb.check_no_breakgraph
def get_fields():
    x = Singleton(1)
    y = Pair(2, 3)
    z = Triplet(4, 5, 6)
    return x._fields + y._fields + z._fields


class TestNamedTuple(TestCaseBase):
    def test_create_namedtuple(self):
        self.assert_results(create_namedtuple)

    def test_create_namedtuple_with_tensor(self):
        x = paddle.to_tensor(1)
        self.assert_results(create_namedtuple_with_tensor, x)

    def test_load_namedtuple(self):
        x = Singleton(paddle.to_tensor(1))
        y = Pair(paddle.to_tensor(2), paddle.to_tensor(3))
        self.assert_results(load_namedtuple, x, y)

    def test_create_namedtuple_with_default(self):
        self.assert_results(create_namedtuple_with_default)

    def test_get_fields(self):
        self.assert_results(get_fields)


if __name__ == "__main__":
    unittest.main()
